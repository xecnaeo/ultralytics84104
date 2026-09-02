#!/usr/bin/env python3
"""Generate paper-oriented P3/P4/P5 SPGM prior-mask previews."""

from __future__ import annotations

import json
import math
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

from ultralytics import YOLO
from ultralytics.nn.modules import ScenePriorGuidedModule


WEIGHTS = Path(
    "/home/liumengdong/xProjects/GP02/yolo26/runs/train/"
    "yolo26s_spgm_center_r0p7_sgd_960_4090_300e_seed0_scalefix/weights/best.pt"
)
IMAGES_DIR = Path("/home/liumengdong/xData/pigdata2025_all/images/test")
LABELS_DIR = Path("/home/liumengdong/xData/pigdata2025_all/labels/test")
OUTPUT_DIR = Path(
    "/home/liumengdong/xProjects/GP02/yolo26/runs/visualize/"
    "spgm_center_r0p7_multiscale_prior_fig4_1"
)
IMGSZ = 960
MAX_CANDIDATES = 12
EXPECTED_SHAPES = {"P3": (120, 120), "P4": (60, 60), "P5": (30, 30)}
COLORMAP = "turbo"
OVERLAY_ALPHA = 0.45


def load_boxes(label_path: Path, width: int, height: int) -> list[tuple[float, float, float, float]]:
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
        boxes.append((x1, y1, bw * width, bh * height))
    return boxes


def scan_dataset() -> list[dict]:
    records = []
    for image_path in sorted(IMAGES_DIR.glob("*")):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        boxes = load_boxes(LABELS_DIR / f"{image_path.stem}.txt", width, height)
        if not boxes:
            continue
        areas = np.array([box[2] * box[3] for box in boxes], dtype=np.float32)
        edge_count = sum(
            x <= 0.05 * width
            or y <= 0.05 * height
            or x + bw >= 0.95 * width
            or y + bh >= 0.95 * height
            for x, y, bw, bh in boxes
        )
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        records.append(
            {
                "image_path": image_path,
                "file_name": image_path.name,
                "width": width,
                "height": height,
                "boxes": boxes,
                "object_count": len(boxes),
                "small_count": int((areas < 32**2).sum()),
                "small_fraction": float((areas < 32**2).mean()),
                "edge_count": int(edge_count),
                "edge_fraction": float(edge_count / len(boxes)),
                "mean_luminance": float(gray.mean()),
                "texture_score": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
                "categories": [],
            }
        )
    return records


def select_candidates(records: list[dict]) -> list[dict]:
    rankings = {
        "dense_small": sorted(
            records,
            key=lambda item: (
                item["small_count"],
                item["small_fraction"],
                item["object_count"],
            ),
            reverse=True,
        ),
        "low_light": sorted(
            [item for item in records if item["object_count"] >= 5],
            key=lambda item: (item["mean_luminance"], -item["object_count"]),
        ),
        "edge": sorted(
            records,
            key=lambda item: (
                item["edge_count"],
                item["edge_fraction"],
                item["object_count"],
            ),
            reverse=True,
        ),
        "complex_bg": sorted(
            [item for item in records if item["object_count"] >= 5],
            key=lambda item: (item["texture_score"], item["object_count"]),
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
        fallback = sorted(
            records,
            key=lambda item: (
                item["object_count"] + 2 * item["small_count"] + item["edge_count"],
                item["texture_score"],
            ),
            reverse=True,
        )
        for item in fallback:
            if item["file_name"] in by_name:
                continue
            item["categories"].append("fallback")
            selected.append(item)
            by_name[item["file_name"]] = item
            if len(selected) == MAX_CANDIDATES:
                break
    return selected


def fixed_letterbox(image: np.ndarray) -> tuple[np.ndarray, dict]:
    height, width = image.shape[:2]
    ratio = min(IMGSZ / height, IMGSZ / width)
    new_width = int(round(width * ratio))
    new_height = int(round(height * ratio))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    dw = IMGSZ - new_width
    dh = IMGSZ - new_height
    left = int(round(dw / 2 - 0.1))
    right = int(round(dw / 2 + 0.1))
    top = int(round(dh / 2 - 0.1))
    bottom = int(round(dh / 2 + 0.1))
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    if padded.shape[:2] != (IMGSZ, IMGSZ):
        raise RuntimeError(f"Unexpected letterbox shape: {padded.shape[:2]}")
    return padded, {
        "ratio": ratio,
        "left": left,
        "right": right,
        "top": top,
        "bottom": bottom,
        "resized_width": new_width,
        "resized_height": new_height,
    }


def restore_mask(mask: np.ndarray, transform: dict, width: int, height: int) -> np.ndarray:
    upsampled = cv2.resize(mask, (IMGSZ, IMGSZ), interpolation=cv2.INTER_LINEAR)
    y1 = transform["top"]
    y2 = IMGSZ - transform["bottom"]
    x1 = transform["left"]
    x2 = IMGSZ - transform["right"]
    cropped = upsampled[y1:y2, x1:x2]
    restored = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)
    if restored.shape != (height, width):
        raise RuntimeError(f"Unexpected restored mask shape: {restored.shape}")
    return np.clip(restored, 0.0, 1.0)


def draw_boxes(axis, boxes, color="cyan", linewidth=1.1) -> None:
    for x, y, width, height in boxes:
        axis.add_patch(
            Rectangle(
                (x, y),
                width,
                height,
                fill=False,
                edgecolor=color,
                linewidth=linewidth,
            )
        )


def show_panel(axis, image_rgb, boxes, mask=None, title="") -> None:
    axis.imshow(image_rgb)
    if mask is not None:
        axis.imshow(
            mask,
            cmap=COLORMAP,
            vmin=0.0,
            vmax=1.0,
            alpha=OVERLAY_ALPHA,
            interpolation="bilinear",
        )
    draw_boxes(axis, boxes)
    axis.set_title(title, fontsize=8)
    axis.axis("off")


def save_candidate_preview(result: dict, preview_dir: Path) -> Path:
    figure, axes = plt.subplots(1, 4, figsize=(12, 2.4), constrained_layout=True)
    show_panel(axes[0], result["image_rgb"], result["boxes"], title="Original + GT")
    for axis, scale in zip(axes[1:], ("P3", "P4", "P5")):
        show_panel(
            axis,
            result["image_rgb"],
            result["boxes"],
            result["restored_masks"][scale],
            f"{scale} prior",
        )
    figure.suptitle(
        f"{result['file_name']} | {', '.join(result['categories'])}",
        fontsize=9,
    )
    path = preview_dir / f"{Path(result['file_name']).stem}_prior_preview.png"
    figure.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def save_contact_sheet(results: list[dict], output_path: Path) -> None:
    candidates_per_row = 3
    row_count = math.ceil(len(results) / candidates_per_row)
    figure = plt.figure(figsize=(22, 3.3 * row_count))
    grid = figure.add_gridspec(row_count, candidates_per_row * 4, wspace=0.03, hspace=0.25)
    for index, result in enumerate(results):
        row = index // candidates_per_row
        start_col = (index % candidates_per_row) * 4
        titles = ("GT", "P3", "P4", "P5")
        masks = (None,) + tuple(result["restored_masks"][scale] for scale in ("P3", "P4", "P5"))
        for panel_index, (title, mask) in enumerate(zip(titles, masks)):
            axis = figure.add_subplot(grid[row, start_col + panel_index])
            show_panel(axis, result["image_rgb"], result["boxes"], mask, title)
            if panel_index == 0:
                axis.text(
                    0.01,
                    -0.08,
                    f"{result['file_name']} | {','.join(result['categories'])}",
                    transform=axis.transAxes,
                    fontsize=6,
                    va="top",
                )
    figure.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def save_recommended_figure(results: list[dict], output_path: Path) -> list[dict]:
    dense = next((item for item in results if "dense_small" in item["categories"]), results[0])
    second = next(
        (
            item
            for category in ("low_light", "edge", "complex_bg")
            for item in results
            if category in item["categories"] and item["file_name"] != dense["file_name"]
        ),
        results[1],
    )
    recommended = [dense, second]

    figure, axes = plt.subplots(2, 4, figsize=(13, 5.6))
    for row, result in enumerate(recommended):
        show_panel(axes[row, 0], result["image_rgb"], result["boxes"], title="Original + GT")
        for col, scale in enumerate(("P3", "P4", "P5"), start=1):
            show_panel(
                axes[row, col],
                result["image_rgb"],
                result["boxes"],
                result["restored_masks"][scale],
                f"{scale} prior",
            )
        axes[row, 0].set_ylabel(result["file_name"], fontsize=8)
    colorbar = figure.colorbar(
        ScalarMappable(norm=Normalize(0.0, 1.0), cmap=COLORMAP),
        ax=axes[:, 1:].ravel().tolist(),
        fraction=0.012,
        pad=0.012,
    )
    colorbar.set_label("Prior probability", fontsize=8)
    figure.subplots_adjust(left=0.04, right=0.94, top=0.95, bottom=0.04, wspace=0.04, hspace=0.14)
    figure.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return recommended


def main() -> None:
    if not WEIGHTS.is_file():
        raise FileNotFoundError(WEIGHTS)
    if not IMAGES_DIR.is_dir() or not LABELS_DIR.is_dir():
        raise FileNotFoundError("PigDetect test images or labels are unavailable")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    candidates = select_candidates(scan_dataset())
    if not candidates:
        raise RuntimeError("No labelled test candidates were found")

    preview_dir = OUTPUT_DIR / "candidate_previews"
    raw_dir = OUTPUT_DIR / "raw_masks"
    source_dir = OUTPUT_DIR / "candidate_images"
    for directory in (OUTPUT_DIR, preview_dir, raw_dir, source_dir):
        directory.mkdir(parents=True, exist_ok=True)

    wrapper = YOLO(str(WEIGHTS))
    model = wrapper.model.to("cuda:0").half().eval()
    modules = [module for module in model.modules() if isinstance(module, ScenePriorGuidedModule)]
    if len(modules) != 3:
        raise RuntimeError(f"Expected 3 SPGM modules, found {len(modules)}")
    for module in modules:
        module.collect_feature_stats = False
        if module.runtime_mode != "normal":
            raise RuntimeError(f"Unexpected SPGM runtime mode: {module.runtime_mode}")

    results = []
    with torch.inference_mode():
        for candidate in candidates:
            image_bgr = cv2.imread(str(candidate["image_path"]))
            padded_bgr, transform = fixed_letterbox(image_bgr)
            tensor = torch.from_numpy(
                np.ascontiguousarray(padded_bgr[:, :, ::-1].transpose(2, 0, 1))
            ).to("cuda:0")
            tensor = tensor.half().div_(255.0).unsqueeze(0)
            model(tensor)
            torch.cuda.synchronize()

            raw_by_area = []
            for module in modules:
                if module.prior_mask is None:
                    raise RuntimeError("SPGM prior_mask was not populated")
                mask = module.prior_mask[0, 0].float().cpu().numpy()
                raw_by_area.append(mask)
            raw_by_area.sort(key=lambda mask: mask.shape[0] * mask.shape[1], reverse=True)
            raw_masks = dict(zip(("P3", "P4", "P5"), raw_by_area))

            restored_masks = {}
            mask_stats = {}
            for scale, mask in raw_masks.items():
                if mask.shape != EXPECTED_SHAPES[scale]:
                    raise RuntimeError(f"{scale} shape {mask.shape}, expected {EXPECTED_SHAPES[scale]}")
                if not np.isfinite(mask).all() or float(mask.min()) < 0.0 or float(mask.max()) > 1.0:
                    raise RuntimeError(f"{scale} contains invalid probability values")
                restored = restore_mask(
                    mask,
                    transform,
                    candidate["width"],
                    candidate["height"],
                )
                restored_masks[scale] = restored
                mask_stats[scale] = {
                    "shape": list(mask.shape),
                    "min": float(mask.min()),
                    "max": float(mask.max()),
                    "mean": float(mask.mean()),
                }

            np.savez_compressed(
                raw_dir / f"{candidate['image_path'].stem}_prior_masks.npz",
                p3_raw=raw_masks["P3"],
                p4_raw=raw_masks["P4"],
                p5_raw=raw_masks["P5"],
                p3_original=restored_masks["P3"],
                p4_original=restored_masks["P4"],
                p5_original=restored_masks["P5"],
            )
            shutil.copy2(candidate["image_path"], source_dir / candidate["file_name"])
            result = {
                **candidate,
                "image_rgb": cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
                "transform": transform,
                "raw_masks": raw_masks,
                "restored_masks": restored_masks,
                "mask_stats": mask_stats,
            }
            result["preview_path"] = save_candidate_preview(result, preview_dir)
            results.append(result)

    contact_sheet = OUTPUT_DIR / "figure4_1_candidate_contact_sheet.png"
    recommended_figure = OUTPUT_DIR / "figure4_1_multiscale_prior_trial.png"
    save_contact_sheet(results, contact_sheet)
    recommended = save_recommended_figure(results, recommended_figure)

    for output_path in (contact_sheet, recommended_figure):
        if cv2.imread(str(output_path)) is None:
            raise RuntimeError(f"Unreadable output image: {output_path}")

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "weights": str(WEIGHTS),
        "input_size": IMGSZ,
        "fixed_probability_range": [0.0, 1.0],
        "colormap": COLORMAP,
        "overlay_alpha": OVERLAY_ALPHA,
        "candidate_count": len(results),
        "recommended_files": [item["file_name"] for item in recommended],
        "contact_sheet": str(contact_sheet),
        "recommended_figure": str(recommended_figure),
        "candidates": [
            {
                "file_name": item["file_name"],
                "categories": item["categories"],
                "object_count": item["object_count"],
                "small_count": item["small_count"],
                "small_fraction": item["small_fraction"],
                "edge_count": item["edge_count"],
                "edge_fraction": item["edge_fraction"],
                "mean_luminance": item["mean_luminance"],
                "texture_score": item["texture_score"],
                "letterbox": item["transform"],
                "mask_stats": item["mask_stats"],
                "preview_path": str(item["preview_path"]),
            }
            for item in results
        ],
    }
    metadata_path = OUTPUT_DIR / "figure4_1_trial_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "status": "ok",
                "candidate_count": len(results),
                "recommended_files": metadata["recommended_files"],
                "contact_sheet": str(contact_sheet),
                "recommended_figure": str(recommended_figure),
                "metadata": str(metadata_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
