"""Train the official YOLOv8s baseline on Server102_4090."""

import hashlib
import subprocess
import sys
from pathlib import Path

import torch
import ultralytics
from ultralytics import YOLO


SOURCE_ROOT = Path("/home/liumengdong/xProjects/GP02/ultralytics84104-yolo26-baseline")
MODEL_YAML = SOURCE_ROOT / "ultralytics/cfg/models/v8/yolov8s.yaml"
OFFICIAL_MODEL_YAML = SOURCE_ROOT / "ultralytics/cfg/models/v8/yolov8.yaml"
DATA_YAML = Path("/home/liumengdong/xProjects/GP01/yolo26/dataset/pigData2025.yaml")
OUTPUT_PROJECT = Path("/home/liumengdong/xProjects/GP02/yolov8/runs/train")
OFFICIAL_BASE_COMMIT = "d8f2cad2ca798701875c5ef91fd5a6a4189781ca"

EPOCHS = 300
BATCH = 16
IMGSZ = 640
DEVICE = "0"
RUN_NAME = "yolov8s_baseline_4090_300e"


def git_output(*args):
    """Run a read-only Git query in the shared baseline worktree."""
    return subprocess.check_output(
        ["git", "-C", str(SOURCE_ROOT), *args],
        text=True,
    ).strip()


def validate_paths():
    """Fail early when source integrity, required paths, or output conditions are invalid."""
    for path in (OFFICIAL_MODEL_YAML, DATA_YAML):
        if not path.is_file():
            raise FileNotFoundError(path)

    imported_root = Path(ultralytics.__file__).resolve().parents[1]
    if imported_root != SOURCE_ROOT:
        raise RuntimeError(f"Unexpected ultralytics source: {ultralytics.__file__}")

    if git_output("rev-parse", "main") != OFFICIAL_BASE_COMMIT:
        raise RuntimeError("Local main no longer matches the pinned official baseline commit.")

    subprocess.run(
        ["git", "-C", str(SOURCE_ROOT), "diff", "--quiet", OFFICIAL_BASE_COMMIT, "HEAD", "--", "ultralytics"],
        check=True,
    )
    if git_output("status", "--porcelain", "--", "ultralytics"):
        raise RuntimeError("The official ultralytics source tree has local changes.")

    run_dir = OUTPUT_PROJECT / RUN_NAME
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists; choose a new RUN_NAME: {run_dir}")


def print_provenance():
    """Print immutable source, model, and runtime provenance into the training log."""
    yaml_sha256 = hashlib.sha256(OFFICIAL_MODEL_YAML.read_bytes()).hexdigest()
    print("\n========== Official Baseline Provenance ==========")
    print("official source commit:", OFFICIAL_BASE_COMMIT)
    print("baseline runner commit:", git_output("rev-parse", "HEAD"))
    print("official model yaml:", OFFICIAL_MODEL_YAML)
    print("official model yaml sha256:", yaml_sha256)
    print("python:", sys.version.replace("\n", " "))
    print("torch:", torch.__version__)
    print("torch cuda:", torch.version.cuda)
    print("ultralytics:", ultralytics.__version__)
    print("ultralytics source:", ultralytics.__file__)
    print("==================================================\n")


def validate_model(model):
    """Confirm the official YOLOv8s topology before training."""
    if model.model.yaml.get("scale") != "s":
        raise RuntimeError(f"Expected YOLOv8s scale, got {model.model.yaml.get('scale')}")
    if any(module.__class__.__name__ == "ScenePriorGuidedModule" for module in model.model.modules()):
        raise RuntimeError("Official baseline unexpectedly contains ScenePriorGuidedModule.")

    detect = model.model.model[-1]
    detect_channels = [branch[0].conv.in_channels for branch in detect.cv2]
    if detect_channels != [128, 256, 512]:
        raise RuntimeError(f"Unexpected Detect channels: {detect_channels}")
    print("official YOLOv8s Detect channels:", detect_channels)


def main():
    """Train the unmodified official YOLOv8s model with the control configuration."""
    validate_paths()
    print_provenance()
    model = YOLO(str(MODEL_YAML))
    validate_model(model)
    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        patience=100,
        batch=BATCH,
        nbs=64,
        imgsz=IMGSZ,
        save=True,
        cache=False,
        device=DEVICE,
        workers=8,
        project=str(OUTPUT_PROJECT),
        name=RUN_NAME,
        exist_ok=False,
        save_period=99,
        pretrained=False,
        amp=True,
        optimizer="AdamW",
        lr0=0.0004,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        close_mosaic=10,
        seed=0,
        deterministic=True,
        multi_scale=0.0,
    )


if __name__ == "__main__":
    main()
