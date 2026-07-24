"""Train the official YOLO26n baseline on Server102_4090."""

from pathlib import Path

from ultralytics import YOLO


SOURCE_ROOT = Path("/home/liumengdong/xProjects/GP02/ultralytics84104-yolo26-baseline")
MODEL_YAML = SOURCE_ROOT / "ultralytics/cfg/models/26/yolo26n.yaml"
OFFICIAL_MODEL_YAML = SOURCE_ROOT / "ultralytics/cfg/models/26/yolo26.yaml"
DATA_YAML = Path("/home/liumengdong/xProjects/GP01/yolo26/dataset/pigData2025.yaml")
OUTPUT_PROJECT = Path("/home/liumengdong/xProjects/GP02/yolo26/runs/train")

EPOCHS = 100
BATCH = 16
IMGSZ = 640
DEVICE = "0"
RUN_NAME = "yolo26n_baseline_4090_100e"


def validate_paths():
    """Fail early when a required model, dataset, or output condition is invalid."""
    for path in (OFFICIAL_MODEL_YAML, DATA_YAML):
        if not path.is_file():
            raise FileNotFoundError(path)

    run_dir = OUTPUT_PROJECT / RUN_NAME
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists; choose a new RUN_NAME: {run_dir}")


def main():
    """Train the unmodified official YOLO26n model with the control configuration."""
    validate_paths()
    model = YOLO(str(MODEL_YAML))
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
    )


if __name__ == "__main__":
    main()
