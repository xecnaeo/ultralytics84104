"""Train YOLO26n-SPGM on Server102_4090 and record auxiliary-prior loss trends."""

import csv
from collections import defaultdict
from pathlib import Path

from ultralytics import YOLO


SOURCE_ROOT = Path("/home/liumengdong/xProjects/GP02/ultralytics84104-spgm-yolo26")
MODEL_YAML = SOURCE_ROOT / "ultralytics/cfg/models/26/yolo26n-SPGM.yaml"
PRETRAINED_WEIGHTS = SOURCE_ROOT / "yolo26n.pt"
DATA_YAML = Path("/home/liumengdong/xProjects/GP01/yolo26/dataset/pigData2025.yaml")
OUTPUT_PROJECT = Path("/home/liumengdong/xProjects/GP02/yolo26/runs/train")

EPOCHS = 300
BATCH = 16
IMGSZ = 640
DEVICE = "0"
RUN_NAME = "yolo26n_spgm_stage6_aux_trend_4090_300e_coco"

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
    """Write single-GPU SPGM auxiliary-prior metrics to per-batch CSV files."""

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


def validate_paths():
    """Fail early when a required model or dataset configuration is missing."""
    for path in (MODEL_YAML, PRETRAINED_WEIGHTS, DATA_YAML):
        if not path.is_file():
            raise FileNotFoundError(path)

    run_dir = OUTPUT_PROJECT / RUN_NAME
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists; choose a new RUN_NAME: {run_dir}")


def load_coco_pretrained_weights(model):
    """Load official YOLO26n weights while accounting for SPGM layers inserted into the head."""
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
    validate_paths()
    model = YOLO(str(MODEL_YAML))
    load_coco_pretrained_weights(model)
    trend_logger = SPGMPriorTrendLogger()
    model.add_callback("on_train_start", trend_logger.on_train_start)
    model.add_callback("on_train_batch_end", trend_logger.on_train_batch_end)
    model.add_callback("on_train_epoch_end", trend_logger.on_train_epoch_end)
    model.add_callback("on_train_end", trend_logger.on_train_end)

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
    )


if __name__ == "__main__":
    main()
