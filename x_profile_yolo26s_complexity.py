#!/usr/bin/env python3
"""Profile one fixed YOLO26 variant on Server102_4090."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


VARIANTS = {
    "baseline": {
        "source_root": Path("/home/liumengdong/xProjects/GP02/ultralytics84104-yolo26-baseline"),
        "weights": Path(
            "/home/liumengdong/xProjects/GP02/yolo26/runs/train/"
            "yolo26s_optimized_sgd_960_4090_300e/weights/best.pt"
        ),
        "output": Path(
            "/home/liumengdong/xProjects/GP02/yolo26/runs/profile/"
            "yolo26s_baseline_960_fp16_b1.json"
        ),
        "expected_spgm_modules": 0,
    },
    "full_spgm": {
        "source_root": Path("/home/liumengdong/xProjects/GP02/ultralytics84104-spgm-yolo26"),
        "weights": Path(
            "/home/liumengdong/xProjects/GP02/yolo26/runs/train/"
            "yolo26s_spgm_center_r0p7_sgd_960_4090_300e_seed0_scalefix/weights/best.pt"
        ),
        "output": Path(
            "/home/liumengdong/xProjects/GP02/yolo26/runs/profile/"
            "yolo26s_spgm_center_r0p7_960_fp16_b1.json"
        ),
        "expected_spgm_modules": 3,
    },
}

IMGSZ = 960
WARMUP = 50
ITERATIONS = 200


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    args = parser.parse_args()
    config = VARIANTS[args.variant]

    source_root = config["source_root"].resolve()
    weights = config["weights"].resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    if not weights.is_file():
        raise FileNotFoundError(weights)

    sys.path.insert(0, str(source_root))

    import torch
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops, get_flops_with_torch_profiler

    import ultralytics

    imported_root = Path(ultralytics.__file__).resolve().parents[1]
    if imported_root != source_root:
        raise RuntimeError(f"Imported {imported_root}, expected {source_root}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    wrapper = YOLO(str(weights))
    model = wrapper.model.eval()
    if args.variant == "full_spgm":
        from ultralytics.nn.modules import ScenePriorGuidedModule

        spgm_modules = [
            module for module in model.modules() if isinstance(module, ScenePriorGuidedModule)
        ]
    else:
        spgm_modules = []
    if len(spgm_modules) != config["expected_spgm_modules"]:
        raise RuntimeError(
            f"Expected {config['expected_spgm_modules']} SPGM modules, found {len(spgm_modules)}"
        )
    for module in spgm_modules:
        module.collect_feature_stats = False
        if module.runtime_mode != "normal":
            raise RuntimeError(f"Unexpected SPGM runtime mode: {module.runtime_mode}")

    params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    gflops = float(get_flops(model, imgsz=IMGSZ))
    if gflops <= 0:
        gflops = float(get_flops_with_torch_profiler(model, imgsz=IMGSZ))

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0")
    model = model.to(device).half().eval()
    sample = torch.randn(1, 3, IMGSZ, IMGSZ, device=device, dtype=torch.float16)

    with torch.inference_mode():
        for _ in range(WARMUP):
            model(sample)
        torch.cuda.synchronize(device)

        elapsed_ms: list[float] = []
        for _ in range(ITERATIONS):
            torch.cuda.synchronize(device)
            started = time.perf_counter_ns()
            model(sample)
            torch.cuda.synchronize(device)
            elapsed_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)

    mean_ms = sum(elapsed_ms) / len(elapsed_ms)
    median_ms = percentile(elapsed_ms, 0.5)
    p95_ms = percentile(elapsed_ms, 0.95)
    result = {
        "variant": args.variant,
        "source_root": str(source_root),
        "source_commit": subprocess.check_output(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
        ).strip(),
        "weights": str(weights),
        "weight_size_bytes": weights.stat().st_size,
        "weight_size_mib": weights.stat().st_size / (1024**2),
        "imgsz": IMGSZ,
        "batch": 1,
        "precision": "FP16",
        "warmup_iterations": WARMUP,
        "timed_iterations": ITERATIONS,
        "scope": "raw model forward; preprocessing and postprocessing excluded",
        "diagnostic_feature_statistics": "disabled",
        "gpu": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "ultralytics_version": ultralytics.__version__,
        "parameters": params,
        "trainable_parameters": trainable_params,
        "gflops": gflops,
        "latency_mean_ms": mean_ms,
        "latency_median_ms": median_ms,
        "latency_p95_ms": p95_ms,
        "fps_from_mean": 1000.0 / mean_ms,
        "spgm_modules": len(spgm_modules),
    }
    output = config["output"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
