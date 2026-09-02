"""Train accuracy-oriented YOLO26s-SPGM on Server102_4090 and record prior-loss trends."""

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import torch


SOURCE_ROOT = Path("/home/liumengdong/xProjects/GP02/ultralytics84104-spgm-yolo26")
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import ultralytics  # noqa: E402
from ultralytics import YOLO  # noqa: E402


MODEL_YAML = SOURCE_ROOT / "ultralytics/cfg/models/26/yolo26s-SPGM.yaml"
OFFICIAL_MODEL_YAML = SOURCE_ROOT / "ultralytics/cfg/models/26/yolo26.yaml"
PRETRAINED_WEIGHTS = Path("/home/liumengdong/xProjects/GP02/yolo26/weights/yolo26s.pt")
DATA_YAML = Path("/home/liumengdong/xProjects/GP01/yolo26/dataset/pigData2025.yaml")
DATASET_LABELS = Path("/home/liumengdong/xData/pigdata2025_all/labels")
OUTPUT_PROJECT = Path("/home/liumengdong/xProjects/GP02/yolo26/runs/train")
OFFICIAL_BASE_COMMIT = "d8f2cad2ca798701875c5ef91fd5a6a4189781ca"

EPOCHS = 300
BATCH = 16
NBS = BATCH
IMGSZ = 960
DEVICE = "0"
LR0 = 0.01 * BATCH / 64
LAMBDA_PRIOR = 0.05
SCALE_WEIGHTS = {"P3": 0.5, "P4": 1.0, "P5": 1.0}
BCE_WEIGHT = 1.0
DICE_WEIGHT = 1.0

COCO_LAYER_MAP = {
    **{index: index for index in range(11)},
    13: 15,
    16: 19,
    17: 20,
    19: 22,
    20: 23,
    22: 25,
    23: 26,
}
EXPECTED_PRETRAINED_ITEMS = 708


class SPGMPriorTrendLogger:
    """Write single-GPU SPGM auxiliary-prior metrics to buffered CSV files."""

    batch_header = [
        "global_step",
        "epoch",
        "batch_i",
        "scale",
        "shape",
        "scale_weight",
        "bce",
        "dice",
        "prior_loss_scale",
        "target_mean",
        "pred_mean",
        "loss_raw",
        "loss_weighted",
        "lambda_prior",
        "mask_mode",
        "center_ratio",
        "num_items",
        "rank",
        "world_size",
    ]
    overall_header = [
        "global_step",
        "epoch",
        "batch_i",
        "loss_raw",
        "loss_weighted",
        "lambda_prior",
        "mask_mode",
        "center_ratio",
        "num_items",
        "rank",
        "world_size",
    ]

    def __init__(self):
        self.global_step = 0
        self.batch_csv = None
        self.overall_csv = None
        self.batch_file = None
        self.overall_file = None
        self.batch_writer = None
        self.overall_writer = None
        self.epoch_losses = defaultdict(list)

    def _flush_csv(self):
        for file in (self.batch_file, self.overall_file):
            if file is not None and not file.closed:
                file.flush()

    def _close_csv(self):
        self._flush_csv()
        for file in (self.batch_file, self.overall_file):
            if file is not None and not file.closed:
                file.close()

    def on_train_start(self, trainer):
        save_dir = Path(trainer.save_dir) / "spgm_prior_trend"
        save_dir.mkdir(parents=True, exist_ok=True)
        self.batch_csv = save_dir / "spgm_prior_batch.csv"
        self.overall_csv = save_dir / "spgm_prior_overall.csv"

        self.batch_file = self.batch_csv.open("w", newline="", encoding="utf-8", buffering=1024 * 1024)
        self.overall_file = self.overall_csv.open("w", newline="", encoding="utf-8", buffering=1024 * 1024)
        self.batch_writer = csv.DictWriter(self.batch_file, fieldnames=self.batch_header)
        self.overall_writer = csv.DictWriter(self.overall_file, fieldnames=self.overall_header)
        self.batch_writer.writeheader()
        self.overall_writer.writeheader()
        self._flush_csv()

        print("\n========== SPGM Prior Trend Logger (single GPU) ==========")
        print("batch csv:", self.batch_csv)
        print("overall csv:", self.overall_csv)
        print("==========================================================\n")

    def on_train_batch_end(self, trainer):
        info = getattr(trainer.model, "spgm_prior_info", None)
        epoch = int(getattr(trainer, "epoch", 0)) + 1
        batch_i = int(getattr(trainer, "batch_i", self.global_step))

        if not info or not info.get("enabled", False):
            if self.global_step == 0:
                print("[SPGM Prior Trend] Warning: no enabled spgm_prior_info was recorded.")
            self.global_step += 1
            return

        common = {
            "global_step": self.global_step,
            "epoch": epoch,
            "batch_i": batch_i,
            "loss_raw": float(info.get("loss_raw", 0.0)),
            "loss_weighted": float(info.get("loss_weighted", 0.0)),
            "lambda_prior": float(info.get("lambda_prior", 0.0)),
            "mask_mode": info.get("mask_mode", ""),
            "center_ratio": float(info.get("center_ratio", 0.0)),
            "num_items": int(info.get("num_items", 0)),
            "rank": -1,
            "world_size": 1,
        }
        self.overall_writer.writerow(common)

        batch_rows = []
        for detail in info.get("detail", []):
            batch_rows.append(
                {
                    **common,
                    "scale": detail.get("scale", ""),
                    "shape": str(tuple(detail.get("shape", ()))),
                    "scale_weight": float(detail.get("scale_weight", 1.0)),
                    "bce": float(detail.get("bce", 0.0)),
                    "dice": float(detail.get("dice", 0.0)),
                    "prior_loss_scale": float(detail.get("loss", 0.0)),
                    "target_mean": float(detail.get("target_mean", 0.0)),
                    "pred_mean": float(detail.get("pred_mean", 0.0)),
                }
            )
        if batch_rows:
            self.batch_writer.writerows(batch_rows)

        self.epoch_losses[epoch].append(common["loss_raw"])
        if self.global_step % 20 == 0:
            print(
                f"\n[SPGM Prior Trend] step={self.global_step}, epoch={epoch}, "
                f"loss_raw={common['loss_raw']:.4f}, "
                f"loss_weighted={common['loss_weighted']:.4f}, "
                f"num_items={common['num_items']}"
            )
        self.global_step += 1

    def on_train_epoch_end(self, trainer):
        self._flush_csv()

    def on_train_end(self, trainer):
        self._close_csv()
        if not self.epoch_losses:
            print("[SPGM Prior Trend] No prior information was recorded.")
            return

        print("\n========== SPGM Prior Trend Summary ==========")
        for epoch, values in sorted(self.epoch_losses.items()):
            print(f"epoch {epoch}: mean loss_raw = {sum(values) / len(values):.4f}")
        print("batch csv:", self.batch_csv)
        print("overall csv:", self.overall_csv)
        print("==============================================\n")


def parse_args():
    """Parse weak-mask, auxiliary-loss, and run-provenance controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask-mode", choices=("binary", "center"), required=True)
    parser.add_argument("--center-ratio", type=float, required=True)
    parser.add_argument("--lambda-prior", type=float, default=LAMBDA_PRIOR)
    parser.add_argument("--modulation-mode", choices=("on", "off"), default="on")
    parser.add_argument("--mosaic-mode", choices=("standard", "off"), default="standard")
    parser.add_argument("--coordinate-mode", choices=("on", "zero"), default="on")
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    if not 0.0 < args.center_ratio <= 1.0:
        parser.error("--center-ratio must be in (0, 1]")
    if args.mask_mode == "binary" and args.center_ratio != 1.0:
        parser.error("binary masks must use --center-ratio 1.0 for unambiguous provenance")
    if args.lambda_prior < 0.0:
        parser.error("--lambda-prior must be non-negative")
    return args


def sha256(path):
    """Return the SHA256 digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(root, pattern="*.txt"):
    """Hash relative paths and contents under a dataset-label directory."""
    digest = hashlib.sha256()
    paths = sorted(root.rglob(pattern))
    if not paths:
        raise FileNotFoundError(f"No {pattern} files found under {root}")
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def build_spgm_aux_config(args):
    """Return the exact auxiliary-loss configuration applied to DetectionModel."""
    return {
        "lambda_prior": args.lambda_prior,
        "scale_weights": SCALE_WEIGHTS.copy(),
        "mask_mode": args.mask_mode,
        "center_ratio": args.center_ratio,
        "bce_weight": BCE_WEIGHT,
        "dice_weight": DICE_WEIGHT,
    }


def build_mosaic_config(args):
    """Return the exact Mosaic schedule passed to Ultralytics."""
    if args.mosaic_mode == "standard":
        return {"mode": "standard", "mosaic": 1.0, "close_mosaic": 20}
    return {"mode": "off", "mosaic": 0.0, "close_mosaic": 0}


def configure_spgm_runtime_mode(model, modulation_mode):
    """Enable normal SPGM modulation or keep only the auxiliary prior branch."""
    runtime_mode = "normal" if modulation_mode == "on" else "identity"
    spgm_modules = [
        module for module in model.modules() if module.__class__.__name__ == "ScenePriorGuidedModule"
    ]
    if len(spgm_modules) != 3:
        raise RuntimeError(f"Expected 3 ScenePriorGuidedModule instances, got {len(spgm_modules)}")
    for module in spgm_modules:
        module.set_runtime_mode(runtime_mode)
    return runtime_mode


def configure_spgm_coordinate_mode(model, coordinate_mode):
    """Use normal coordinates or zero them while preserving SPGM capacity."""
    runtime_mode = "normal" if coordinate_mode == "on" else "zero"
    spgm_modules = [
        module for module in model.modules() if module.__class__.__name__ == "ScenePriorGuidedModule"
    ]
    if len(spgm_modules) != 3:
        raise RuntimeError(f"Expected 3 ScenePriorGuidedModule instances, got {len(spgm_modules)}")
    for module in spgm_modules:
        module.set_coordinate_mode(runtime_mode)
    return runtime_mode


class SPGMExperimentConfigurator:
    """Apply SPGM loss controls to the trainer model and persist experiment provenance."""

    def __init__(self, args):
        self.args = args
        self.spgm_aux_config = build_spgm_aux_config(args)

    def on_train_start(self, trainer):
        trainer.model.spgm_aux_config = self.spgm_aux_config.copy()
        runtime_mode = configure_spgm_runtime_mode(trainer.model, self.args.modulation_mode)
        coordinate_runtime_mode = configure_spgm_coordinate_mode(
            trainer.model, self.args.coordinate_mode
        )
        mosaic_config = build_mosaic_config(self.args)
        record = {
            "official_source_commit": OFFICIAL_BASE_COMMIT,
            "spgm_source_commit": git_output("rev-parse", "HEAD"),
            "model_yaml": str(MODEL_YAML),
            "model_yaml_sha256": sha256(MODEL_YAML),
            "pretrained_weights": str(PRETRAINED_WEIGHTS),
            "pretrained_weights_sha256": sha256(PRETRAINED_WEIGHTS),
            "data_yaml": str(DATA_YAML),
            "data_yaml_sha256": sha256(DATA_YAML),
            "dataset_label_sha256": {
                split: sha256_tree(DATASET_LABELS / split) for split in ("train", "val", "test")
            },
            "spgm_aux_config": self.spgm_aux_config,
            "modulation_mode": self.args.modulation_mode,
            "spgm_runtime_mode": runtime_mode,
            "coordinate_mode": self.args.coordinate_mode,
            "spgm_coordinate_mode": coordinate_runtime_mode,
            "training": {
                "epochs": EPOCHS,
                "batch": BATCH,
                "nbs": NBS,
                "imgsz": IMGSZ,
                "device": DEVICE,
                "lr0": LR0,
                "seed": 0,
                "deterministic": True,
                "mosaic_mode": mosaic_config["mode"],
                "mosaic": mosaic_config["mosaic"],
                "close_mosaic": mosaic_config["close_mosaic"],
            },
            "run_name": self.args.run_name,
        }
        record_path = Path(trainer.save_dir) / "spgm_experiment.json"
        record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print("SPGM auxiliary config:", self.spgm_aux_config)
        print("SPGM experiment record:", record_path)


def git_output(*args):
    """Run a read-only Git query in the YOLO26-SPGM worktree."""
    return subprocess.check_output(["git", "-C", str(SOURCE_ROOT), *args], text=True).strip()


def validate_paths(args):
    """Fail early when source integrity, required inputs, or output conditions are invalid."""
    for path in (MODEL_YAML, OFFICIAL_MODEL_YAML, PRETRAINED_WEIGHTS, DATA_YAML, DATASET_LABELS):
        if not path.exists():
            raise FileNotFoundError(path)

    imported_root = Path(ultralytics.__file__).resolve().parents[1]
    if imported_root != SOURCE_ROOT:
        raise RuntimeError(f"Unexpected ultralytics source: {ultralytics.__file__}")
    if git_output("rev-parse", "main") != OFFICIAL_BASE_COMMIT:
        raise RuntimeError("Local main no longer matches the pinned official source commit.")
    if git_output("status", "--porcelain", "--", "ultralytics"):
        raise RuntimeError("The SPGM ultralytics source tree has local changes.")

    run_dir = OUTPUT_PROJECT / args.run_name
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists; choose a new --run-name: {run_dir}")


def print_provenance(args):
    """Print source, model, pretrained-weight, and runtime provenance into the training log."""
    print("\n========== Optimized YOLO26s-SPGM Provenance ==========")
    print("official source commit:", OFFICIAL_BASE_COMMIT)
    print("SPGM worktree commit:", git_output("rev-parse", "HEAD"))
    print("SPGM model yaml:", MODEL_YAML)
    print("SPGM model yaml sha256:", sha256(MODEL_YAML))
    print("official model yaml:", OFFICIAL_MODEL_YAML)
    print("official model yaml sha256:", sha256(OFFICIAL_MODEL_YAML))
    print("COCO pretrained weights:", PRETRAINED_WEIGHTS)
    print("COCO pretrained weights sha256:", sha256(PRETRAINED_WEIGHTS))
    print("data yaml:", DATA_YAML)
    print("data yaml sha256:", sha256(DATA_YAML))
    print("SPGM auxiliary config:", build_spgm_aux_config(args))
    print("SPGM modulation mode:", args.modulation_mode)
    print("SPGM coordinate mode:", args.coordinate_mode)
    print("Mosaic config:", build_mosaic_config(args))
    print("run name:", args.run_name)
    print("python:", sys.version.replace("\n", " "))
    print("torch:", torch.__version__)
    print("torch cuda:", torch.version.cuda)
    print("ultralytics:", ultralytics.__version__)
    print("ultralytics source:", ultralytics.__file__)
    print("=======================================================\n")


def validate_model(model):
    """Confirm the model is the expected three-scale YOLO26s-SPGM detector."""
    if model.task != "detect":
        raise RuntimeError(f"Expected a detection model, got task={model.task}")
    if model.model.yaml.get("scale") != "s":
        raise RuntimeError(f"Expected YOLO26s scale, got {model.model.yaml.get('scale')}")

    parameter_count = sum(parameter.numel() for parameter in model.model.parameters())
    if parameter_count != 10_112_196:
        raise RuntimeError(f"Unexpected YOLO26s-SPGM parameter count: {parameter_count}")

    spgm_modules = [
        module for module in model.model.modules() if module.__class__.__name__ == "ScenePriorGuidedModule"
    ]
    if len(spgm_modules) != 3:
        raise RuntimeError(f"Expected 3 ScenePriorGuidedModule instances, got {len(spgm_modules)}")
    if any(not module.collect_aux_logits for module in spgm_modules):
        raise RuntimeError("All SPGM modules must collect auxiliary logits.")
    if any(module.modulation_type != "fg_bg" for module in spgm_modules):
        raise RuntimeError("Unexpected SPGM modulation type.")

    detect = model.model.model[-1]
    detect_channels = [branch[0].conv.in_channels for branch in detect.cv2]
    if detect_channels != [128, 256, 512]:
        raise RuntimeError(f"Unexpected Detect channels: {detect_channels}")
    print("official YOLO26s-SPGM Detect channels:", detect_channels)


def load_coco_pretrained_weights(model):
    """Load official YOLO26s weights while accounting for SPGM layers inserted into the head."""
    initial_state = {name: value.detach().clone() for name, value in model.model.state_dict().items()}
    model.load(str(PRETRAINED_WEIGHTS))
    source_model = model.ckpt.get("ema")
    if source_model is None:
        source_model = model.ckpt["model"]
    source_state = source_model.float().state_dict()

    model.model.load_state_dict(initial_state, strict=True)
    target_state = model.model.state_dict()
    remapped_state = {}
    for source_name, value in source_state.items():
        prefix, layer_text, suffix = source_name.split(".", 2)
        target_layer = COCO_LAYER_MAP.get(int(layer_text))
        if target_layer is None:
            continue
        target_name = f"{prefix}.{target_layer}.{suffix}"
        if target_name in target_state and target_state[target_name].shape == value.shape:
            remapped_state[target_name] = value

    if len(remapped_state) != EXPECTED_PRETRAINED_ITEMS:
        raise RuntimeError(
            f"Expected {EXPECTED_PRETRAINED_ITEMS} compatible COCO state items, got {len(remapped_state)}"
        )
    model.model.load_state_dict(remapped_state, strict=False)
    print(f"Remapped {len(remapped_state)}/{len(target_state)} state items from {PRETRAINED_WEIGHTS}")


def main():
    """Fine-tune COCO-pretrained YOLO26s-SPGM with baseline-matched optimized settings."""
    args = parse_args()
    validate_paths(args)
    print_provenance(args)
    model = YOLO(str(MODEL_YAML))
    validate_model(model)
    load_coco_pretrained_weights(model)
    model.model.spgm_aux_config = build_spgm_aux_config(args)
    configure_spgm_runtime_mode(model.model, args.modulation_mode)
    configure_spgm_coordinate_mode(model.model, args.coordinate_mode)
    mosaic_config = build_mosaic_config(args)

    experiment_configurator = SPGMExperimentConfigurator(args)
    trend_logger = SPGMPriorTrendLogger()
    model.add_callback("on_train_start", experiment_configurator.on_train_start)
    model.add_callback("on_train_start", trend_logger.on_train_start)
    model.add_callback("on_train_batch_end", trend_logger.on_train_batch_end)
    model.add_callback("on_train_epoch_end", trend_logger.on_train_epoch_end)
    model.add_callback("on_train_end", trend_logger.on_train_end)

    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        patience=0,
        batch=BATCH,
        nbs=NBS,
        imgsz=IMGSZ,
        save=True,
        save_period=50,
        cache=False,
        device=DEVICE,
        workers=8,
        project=str(OUTPUT_PROJECT),
        name=args.run_name,
        exist_ok=False,
        pretrained=True,
        resume=False,
        amp=True,
        optimizer="SGD",
        lr0=LR0,
        lrf=0.01,
        cos_lr=False,
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
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=mosaic_config["mosaic"],
        mixup=0.0,
        cutmix=0.0,
        copy_paste=0.0,
        close_mosaic=mosaic_config["close_mosaic"],
        rect=False,
        multi_scale=0.0,
        val=True,
        iou=0.75,
        max_det=100,
        plots=True,
        seed=0,
        deterministic=True,
    )


if __name__ == "__main__":
    main()
