"""Train YOLO26n-SPGM on Server102_4090 and record auxiliary-prior loss trends."""

import csv
from collections import defaultdict
from pathlib import Path

from ultralytics import YOLO


SOURCE_ROOT = Path("/home/liumengdong/xProjects/GP02/ultralytics84104-spgm-yolo26")
MODEL_YAML = SOURCE_ROOT / "ultralytics/cfg/models/26/yolo26n-SPGM.yaml"
DATA_YAML = Path("/home/liumengdong/xProjects/GP01/yolo26/dataset/pigData2025.yaml")
OUTPUT_PROJECT = Path("/home/liumengdong/xProjects/GP02/yolo26/runs/train")

EPOCHS = 100
BATCH = 16
IMGSZ = 640
DEVICE = "0"
RUN_NAME = "yolo26n_spgm_stage6_aux_trend_4090_100e"


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
        self.epoch_losses = defaultdict(list)

    @staticmethod
    def _append_rows(path, fieldnames, rows):
        with path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writerows(rows)

    def on_train_start(self, trainer):
        save_dir = Path(trainer.save_dir) / "spgm_prior_trend"
        save_dir.mkdir(parents=True, exist_ok=True)
        self.batch_csv = save_dir / "spgm_prior_batch.csv"
        self.overall_csv = save_dir / "spgm_prior_overall.csv"

        for path, header in (
            (self.batch_csv, self.batch_header),
            (self.overall_csv, self.overall_header),
        ):
            with path.open("w", newline="", encoding="utf-8") as file:
                csv.DictWriter(file, fieldnames=header).writeheader()

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
        self._append_rows(self.overall_csv, self.overall_header, [common])

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
            self._append_rows(self.batch_csv, self.batch_header, batch_rows)

        self.epoch_losses[epoch].append(common["loss_raw"])
        if self.global_step % 20 == 0:
            print(
                f"\n[SPGM Prior Trend] step={self.global_step}, epoch={epoch}, "
                f"loss_raw={common['loss_raw']:.4f}, "
                f"loss_weighted={common['loss_weighted']:.4f}, "
                f"num_items={common['num_items']}"
            )
        self.global_step += 1

    def on_train_end(self, trainer):
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
    for path in (MODEL_YAML, DATA_YAML):
        if not path.is_file():
            raise FileNotFoundError(path)

    run_dir = OUTPUT_PROJECT / RUN_NAME
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists; choose a new RUN_NAME: {run_dir}")


def main():
    validate_paths()
    model = YOLO(str(MODEL_YAML))
    trend_logger = SPGMPriorTrendLogger()
    model.add_callback("on_train_start", trend_logger.on_train_start)
    model.add_callback("on_train_batch_end", trend_logger.on_train_batch_end)
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
    )


if __name__ == "__main__":
    main()