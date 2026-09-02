#!/usr/bin/env python3
"""Create baseline-vs-SPGM detection comparison candidates from saved predictions."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


BASELINE_JSON = Path(
    "/home/liumengdong/xProjects/GP02/yolo26/runs/val/"
    "yolo26s_optimized_sgd_960_4090_300e_test_cocoeval/predictions.json"
)
SPGM_JSON = Path(
    "/home/liumengdong/xProjects/GP02/yolo26/runs/val/"
    "yolo26s_spgm_center_r0p7_sgd_960_4090_300e_seed0_scalefix_test_cocoeval/"
    "predictions.json"
)
IMAGES_DIR = Path("/home/liumengdong/xData/pigdata2025_all/images/test")
LABELS_DIR = Path("/home/liumengdong/xData/pigdata2025_all/labels/test")
OUTPUT_DIR = Path(
    "/home/liumengdong/xProjects/GP02/yolo26/runs/visualize/"
    "yolo26s_vs_spgm_detection_fig4_2"
)
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.50
MAX_CANDIDATES = 12


def load_prediction_groups(path: Path) -> dict[str, list[dict]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    groups = defaultdict(list)
    for item in json.loads(path.read_text(encoding="utf-8")):
        file_name = item.get("file_name")
        if not file_name:
            image_id = str(item["image_id"])
            file_name = image_id if Path(image_id).suffix else f"{image_id}.jpg"
        if float(item["score"]) < CONF_THRESHOLD:
            continue
        x, y, width, height = map(float, item["bbox"])
        groups[file_name].append(
            {
                "box": (x, y, x + width, y + height),
                "score": float(item["score"]),
            }
        )
    for predictions in groups.values():
        predictions.sort(key=lambda item: item["score"], reverse=True)
    return dict(groups)


def load_gt(label_path: Path, width: int, height: int) -> list[tuple[float, float, float, float]]:
    boxes = []
    if not label_path.is_file():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if len(values) < 5:
            continue
        _, xc, yc, bw, bh = map(float, values[:5])
        x1 = (xc - bw / 2) * width
        y1 = (yc - bh / 2) * height
        x2 = (xc + bw / 2) * width
        y2 = (yc + bh / 2) * height
        boxes.append((x1, y1, x2, y2))
    return boxes


def box_iou(box_a, box_b) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def match_predictions(gt_boxes, predictions) -> dict:
    unmatched_gt = set(range(len(gt_boxes)))
    matched_prediction_indices = set()
    matched_gt_indices = set()
    matches = []
    for prediction_index, prediction in enumerate(predictions):
        best_gt = None
        best_iou = 0.0
        for gt_index in unmatched_gt:
            iou = box_iou(gt_boxes[gt_index], prediction["box"])
            if iou > best_iou:
                best_iou = iou
                best_gt = gt_index
        if best_gt is not None and best_iou >= IOU_THRESHOLD:
            unmatched_gt.remove(best_gt)
            matched_gt_indices.add(best_gt)
            matched_prediction_indices.add(prediction_index)
            matches.append(
                {
                    "prediction_index": prediction_index,
                    "gt_index": best_gt,
                    "iou": best_iou,
                }
            )
    small_gt_indices = {
        index
        for index, box in enumerate(gt_boxes)
        if (box[2] - box[0]) * (box[3] - box[1]) < 32**2
    }
    return {
        "tp": len(matches),
        "fp": len(predictions) - len(matches),
        "fn": len(gt_boxes) - len(matches),
        "small_tp": len(matched_gt_indices & small_gt_indices),
        "small_fn": len(small_gt_indices - matched_gt_indices),
        "matched_prediction_indices": matched_prediction_indices,
        "matched_gt_indices": matched_gt_indices,
        "matches": matches,
    }


def evaluate_images(baseline_groups, spgm_groups) -> list[dict]:
    records = []
    for image_path in sorted(IMAGES_DIR.glob("*")):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        gt_boxes = load_gt(LABELS_DIR / f"{image_path.stem}.txt", width, height)
        if not gt_boxes:
            continue
        baseline_predictions = baseline_groups.get(image_path.name, [])
        spgm_predictions = spgm_groups.get(image_path.name, [])
        baseline_eval = match_predictions(gt_boxes, baseline_predictions)
        spgm_eval = match_predictions(gt_boxes, spgm_predictions)
        record = {
            "image_path": image_path,
            "file_name": image_path.name,
            "image_rgb": cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            "width": width,
            "height": height,
            "gt_boxes": gt_boxes,
            "baseline_predictions": baseline_predictions,
            "spgm_predictions": spgm_predictions,
            "baseline_eval": baseline_eval,
            "spgm_eval": spgm_eval,
            "object_count": len(gt_boxes),
            "tp_gain": spgm_eval["tp"] - baseline_eval["tp"],
            "fn_reduction": baseline_eval["fn"] - spgm_eval["fn"],
            "fp_reduction": baseline_eval["fp"] - spgm_eval["fp"],
            "small_tp_gain": spgm_eval["small_tp"] - baseline_eval["small_tp"],
            "categories": [],
        }
        record["aggregate_gain"] = (
            4 * record["small_tp_gain"]
            + 3 * record["fn_reduction"]
            + max(0, record["fp_reduction"])
        )
        records.append(record)
    return records


def select_candidates(records: list[dict]) -> list[dict]:
    rankings = {
        "small_recovery": sorted(
            [item for item in records if item["small_tp_gain"] > 0],
            key=lambda item: (
                item["small_tp_gain"],
                item["fn_reduction"],
                item["fp_reduction"],
            ),
            reverse=True,
        ),
        "fewer_misses": sorted(
            [item for item in records if item["fn_reduction"] > 0],
            key=lambda item: (
                item["fn_reduction"],
                item["small_tp_gain"],
                item["fp_reduction"],
            ),
            reverse=True,
        ),
        "fewer_fp": sorted(
            [
                item
                for item in records
                if item["fp_reduction"] > 0 and item["spgm_eval"]["tp"] >= item["baseline_eval"]["tp"]
            ],
            key=lambda item: (
                item["fp_reduction"],
                item["fn_reduction"],
                item["small_tp_gain"],
            ),
            reverse=True,
        ),
        "dense_gain": sorted(
            [
                item
                for item in records
                if item["object_count"] >= 15 and item["aggregate_gain"] > 0
            ],
            key=lambda item: (
                item["aggregate_gain"],
                item["object_count"],
                item["fp_reduction"],
            ),
            reverse=True,
        ),
    }

    selected = []
    by_name = {}
    for category, ranked in rankings.items():
        added = 0
        for item in ranked:
            if item["file_name"] in by_name:
                if category not in by_name[item["file_name"]]["categories"]:
                    by_name[item["file_name"]]["categories"].append(category)
                continue
            item["categories"].append(category)
            selected.append(item)
            by_name[item["file_name"]] = item
            added += 1
            if added == 3 or len(selected) == MAX_CANDIDATES:
                break
        if len(selected) == MAX_CANDIDATES:
            break

    if len(selected) < MAX_CANDIDATES:
        positive = sorted(
            [item for item in records if item["aggregate_gain"] > 0],
            key=lambda item: (
                item["aggregate_gain"],
                item["small_tp_gain"],
                item["fp_reduction"],
            ),
            reverse=True,
        )
        for item in positive:
            if item["file_name"] in by_name:
                continue
            item["categories"].append("overall_gain")
            selected.append(item)
            by_name[item["file_name"]] = item
            if len(selected) == MAX_CANDIDATES:
                break
    if not selected:
        raise RuntimeError("No positive comparison candidates found at the fixed thresholds")
    return selected


def draw_gt(axis, gt_boxes, linewidth=0.8, alpha=0.75) -> None:
    for x1, y1, x2, y2 in gt_boxes:
        axis.add_patch(
            Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                edgecolor="cyan",
                linewidth=linewidth,
                linestyle="--",
                alpha=alpha,
            )
        )


def draw_predictions(axis, predictions, evaluation) -> None:
    matched = evaluation["matched_prediction_indices"]
    for index, prediction in enumerate(predictions):
        x1, y1, x2, y2 = prediction["box"]
        color = "lime" if index in matched else "red"
        axis.add_patch(
            Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                edgecolor=color,
                linewidth=1.2,
            )
        )
        axis.text(
            x1,
            max(4.0, y1 - 2.0),
            f"{prediction['score']:.2f}",
            color=color,
            fontsize=5,
            bbox={"facecolor": "black", "alpha": 0.45, "pad": 0.5, "edgecolor": "none"},
        )


def show_original(axis, record, title="Ground truth") -> None:
    axis.imshow(record["image_rgb"])
    draw_gt(axis, record["gt_boxes"], linewidth=1.1, alpha=1.0)
    axis.set_title(title, fontsize=8)
    axis.axis("off")


def show_detection(axis, record, variant, title) -> None:
    axis.imshow(record["image_rgb"])
    draw_gt(axis, record["gt_boxes"])
    predictions = record[f"{variant}_predictions"]
    evaluation = record[f"{variant}_eval"]
    draw_predictions(axis, predictions, evaluation)
    axis.set_title(
        f"{title} | TP {evaluation['tp']}  FP {evaluation['fp']}  FN {evaluation['fn']}",
        fontsize=8,
    )
    axis.axis("off")


def save_preview(record: dict, preview_dir: Path) -> Path:
    figure, axes = plt.subplots(1, 3, figsize=(9.3, 2.4), constrained_layout=True)
    show_original(axes[0], record)
    show_detection(axes[1], record, "baseline", "YOLO26s")
    show_detection(axes[2], record, "spgm", "YOLO26s-SPGM")
    figure.suptitle(
        (
            f"{record['file_name']} | {','.join(record['categories'])} | "
            f"smallTP {record['small_tp_gain']:+d}, FN {record['fn_reduction']:+d}, "
            f"FP {record['fp_reduction']:+d}"
        ),
        fontsize=8,
    )
    path = preview_dir / f"{record['image_path'].stem}_comparison_preview.png"
    figure.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def save_contact_sheet(records: list[dict], output_path: Path) -> None:
    candidates_per_row = 3
    row_count = math.ceil(len(records) / candidates_per_row)
    figure = plt.figure(figsize=(18, 3.3 * row_count))
    grid = figure.add_gridspec(row_count, candidates_per_row * 3, wspace=0.03, hspace=0.25)
    for index, record in enumerate(records):
        row = index // candidates_per_row
        start_col = (index % candidates_per_row) * 3
        axes = [figure.add_subplot(grid[row, start_col + offset]) for offset in range(3)]
        show_original(axes[0], record, "GT")
        show_detection(axes[1], record, "baseline", "Baseline")
        show_detection(axes[2], record, "spgm", "SPGM")
        axes[0].text(
            0.01,
            -0.08,
            (
                f"{record['file_name']} | {','.join(record['categories'])} | "
                f"small {record['small_tp_gain']:+d}, FN {record['fn_reduction']:+d}, "
                f"FP {record['fp_reduction']:+d}"
            ),
            transform=axes[0].transAxes,
            fontsize=5.5,
            va="top",
        )
    figure.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def choose_recommended(records: list[dict]) -> list[dict]:
    preferred_names = ("danuma_1346.jpg", "danuma_1363.jpg", "danuma_1349.jpg")
    by_name = {item["file_name"]: item for item in records}
    preferred = [by_name[name] for name in preferred_names if name in by_name]
    if len(preferred) == 3:
        return preferred

    recommended = []
    for category in ("small_recovery", "fewer_misses", "fewer_fp", "dense_gain", "overall_gain"):
        candidate = next(
            (
                item
                for item in records
                if category in item["categories"] and item not in recommended
            ),
            None,
        )
        if candidate is not None:
            recommended.append(candidate)
        if len(recommended) == 3:
            break
    for item in records:
        if item not in recommended:
            recommended.append(item)
        if len(recommended) == min(3, len(records)):
            break
    return recommended


def save_trial_figure(records: list[dict], output_path: Path) -> list[dict]:
    recommended = choose_recommended(records)
    figure, axes = plt.subplots(len(recommended), 3, figsize=(11.5, 3.15 * len(recommended)))
    if len(recommended) == 1:
        axes = np.expand_dims(axes, axis=0)
    for row, record in enumerate(recommended):
        show_original(axes[row, 0], record)
        show_detection(axes[row, 1], record, "baseline", "YOLO26s")
        show_detection(axes[row, 2], record, "spgm", "YOLO26s-SPGM")
        axes[row, 0].set_ylabel(record["file_name"], fontsize=8)
    legend = [
        Line2D([0], [0], color="cyan", linestyle="--", label="Ground truth"),
        Line2D([0], [0], color="lime", label="Matched detection"),
        Line2D([0], [0], color="red", label="False positive"),
    ]
    figure.legend(handles=legend, loc="lower center", ncol=3, fontsize=8)
    figure.subplots_adjust(left=0.05, right=0.99, top=0.97, bottom=0.06, wspace=0.03, hspace=0.12)
    figure.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return recommended


def serializable_record(record: dict) -> dict:
    return {
        "file_name": record["file_name"],
        "categories": record["categories"],
        "object_count": record["object_count"],
        "tp_gain": record["tp_gain"],
        "fn_reduction": record["fn_reduction"],
        "fp_reduction": record["fp_reduction"],
        "small_tp_gain": record["small_tp_gain"],
        "aggregate_gain": record["aggregate_gain"],
        "baseline": {
            key: record["baseline_eval"][key]
            for key in ("tp", "fp", "fn", "small_tp", "small_fn")
        },
        "spgm": {
            key: record["spgm_eval"][key]
            for key in ("tp", "fp", "fn", "small_tp", "small_fn")
        },
        "preview_path": str(record["preview_path"]),
    }


def main() -> None:
    baseline_groups = load_prediction_groups(BASELINE_JSON)
    spgm_groups = load_prediction_groups(SPGM_JSON)
    records = evaluate_images(baseline_groups, spgm_groups)
    candidates = select_candidates(records)

    preview_dir = OUTPUT_DIR / "candidate_previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    for record in candidates:
        record["preview_path"] = save_preview(record, preview_dir)

    contact_sheet = OUTPUT_DIR / "figure4_2_candidate_contact_sheet.png"
    trial_figure = OUTPUT_DIR / "figure4_2_detection_comparison_trial.png"
    save_contact_sheet(candidates, contact_sheet)
    recommended = save_trial_figure(candidates, trial_figure)

    for output_path in (contact_sheet, trial_figure):
        image = cv2.imread(str(output_path))
        if image is None or image.size == 0:
            raise RuntimeError(f"Unreadable output image: {output_path}")

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "baseline_predictions": str(BASELINE_JSON),
        "spgm_predictions": str(SPGM_JSON),
        "confidence_threshold": CONF_THRESHOLD,
        "iou_threshold": IOU_THRESHOLD,
        "candidate_count": len(candidates),
        "recommended_files": [record["file_name"] for record in recommended],
        "contact_sheet": str(contact_sheet),
        "trial_figure": str(trial_figure),
        "candidates": [serializable_record(record) for record in candidates],
    }
    metadata_path = OUTPUT_DIR / "figure4_2_trial_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "candidate_count": len(candidates),
                "recommended_files": metadata["recommended_files"],
                "contact_sheet": str(contact_sheet),
                "trial_figure": str(trial_figure),
                "metadata": str(metadata_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
