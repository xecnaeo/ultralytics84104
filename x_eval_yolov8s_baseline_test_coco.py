"""Evaluate the official YOLOv8s baseline on the test split with COCOeval."""

import hashlib
import json
import subprocess
from importlib.metadata import version
from pathlib import Path

import ultralytics
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO


SOURCE_ROOT = Path("/home/liumengdong/xProjects/GP02/ultralytics84104-yolo26-baseline")
WEIGHTS = Path(
    "/home/liumengdong/xProjects/GP02/yolov8/runs/train/"
    "yolov8s_baseline_4090_300e/weights/best.pt"
)
DATA_YAML = Path("/home/liumengdong/xProjects/GP01/yolo26/dataset/pigData2025.yaml")
DATA_ROOT = Path("/home/liumengdong/xData/pigdata2025_all")
TEST_IMAGES = DATA_ROOT / "images/test"
TEST_LABELS = DATA_ROOT / "labels/test"
OUTPUT_PROJECT = Path("/home/liumengdong/xProjects/GP02/yolov8/runs/val")
RUN_NAME = "yolov8s_baseline_4090_300e_test_cocoeval"
RUN_DIR = OUTPUT_PROJECT / RUN_NAME
OFFICIAL_BASE_COMMIT = "d8f2cad2ca798701875c5ef91fd5a6a4189781ca"
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def sha256(path):
    """Return the SHA256 digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(*args):
    """Run a read-only Git query in the shared baseline worktree."""
    return subprocess.check_output(["git", "-C", str(SOURCE_ROOT), *args], text=True).strip()


def validate_inputs():
    """Fail early when source integrity, inputs, or output conditions are invalid."""
    for path in (WEIGHTS, DATA_YAML, TEST_IMAGES, TEST_LABELS):
        if not path.exists():
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
    if RUN_DIR.exists():
        raise FileExistsError(f"Evaluation directory already exists: {RUN_DIR}")


def build_coco_ground_truth(output_path):
    """Convert the test split's YOLO labels to a deterministic COCO ground-truth JSON."""
    image_paths = sorted(path for path in TEST_IMAGES.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    labels_by_stem = {path.stem: path for path in TEST_LABELS.glob("*.txt")}
    if not image_paths:
        raise RuntimeError(f"No test images found in {TEST_IMAGES}")
    if len({path.stem for path in image_paths}) != len(image_paths):
        raise RuntimeError("Test image stems are not unique.")

    image_stems = {path.stem for path in image_paths}
    if image_stems != set(labels_by_stem):
        raise RuntimeError(
            f"Image/label stem mismatch: images_only={sorted(image_stems - set(labels_by_stem))[:5]}, "
            f"labels_only={sorted(set(labels_by_stem) - image_stems)[:5]}"
        )

    images = []
    annotations = []
    filename_to_id = {}
    annotation_id = 1
    for image_id, image_path in enumerate(image_paths, 1):
        with Image.open(image_path) as image:
            width, height = image.size
        filename_to_id[image_path.name] = image_id
        images.append({"id": image_id, "file_name": image_path.name, "width": width, "height": height})

        for line_number, line in enumerate(labels_by_stem[image_path.stem].read_text().splitlines(), 1):
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(f"Invalid label at {labels_by_stem[image_path.stem]}:{line_number}")
            class_id, center_x, center_y, box_width, box_height = map(float, fields)
            if class_id != 0:
                raise ValueError(f"Unexpected class {class_id} at {labels_by_stem[image_path.stem]}:{line_number}")
            if not (
                0 <= center_x - box_width / 2 <= center_x + box_width / 2 <= 1
                and 0 <= center_y - box_height / 2 <= center_y + box_height / 2 <= 1
                and box_width > 0
                and box_height > 0
            ):
                raise ValueError(f"Out-of-bounds box at {labels_by_stem[image_path.stem]}:{line_number}")

            x = (center_x - box_width / 2) * width
            y = (center_y - box_height / 2) * height
            w = box_width * width
            h = box_height * height
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    ground_truth = {
        "info": {"description": "pigData2025 test split converted from YOLO labels", "version": "1.0"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "pig", "supercategory": "animal"}],
    }
    output_path.write_text(json.dumps(ground_truth, indent=2), encoding="utf-8")
    return filename_to_id, len(images), len(annotations)


def normalize_predictions(raw_path, output_path, filename_to_id):
    """Replace Ultralytics string stems with deterministic integer COCO image IDs."""
    raw_predictions = json.loads(raw_path.read_text(encoding="utf-8"))
    normalized = []
    for prediction in raw_predictions:
        file_name = prediction.get("file_name")
        if file_name not in filename_to_id:
            raise KeyError(f"Prediction references unknown image: {file_name}")
        if prediction["category_id"] != 1:
            raise ValueError(f"Unexpected prediction category: {prediction['category_id']}")
        normalized.append(
            {
                "image_id": filename_to_id[file_name],
                "category_id": 1,
                "bbox": prediction["bbox"],
                "score": prediction["score"],
            }
        )
    output_path.write_text(json.dumps(normalized, indent=2), encoding="utf-8")
    return len(normalized)


def run_coco_eval(ground_truth_path, predictions_path):
    """Run standard pycocotools COCOeval and return all 12 summary metrics."""
    coco_ground_truth = COCO(str(ground_truth_path))
    coco_predictions = coco_ground_truth.loadRes(str(predictions_path))
    evaluator = COCOeval(coco_ground_truth, coco_predictions, "bbox")
    evaluator.params.imgIds = sorted(coco_ground_truth.getImgIds())
    evaluator.params.catIds = [1]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    names = [
        "AP_50_95",
        "AP_50",
        "AP_75",
        "AP_small",
        "AP_medium",
        "AP_large",
        "AR_1",
        "AR_10",
        "AR_100",
        "AR_small",
        "AR_medium",
        "AR_large",
    ]
    return dict(zip(names, map(float, evaluator.stats)))


def main():
    """Export test predictions and evaluate them with standard COCOeval."""
    validate_inputs()
    print("best weights:", WEIGHTS)
    print("best weights sha256:", sha256(WEIGHTS))
    print("ultralytics:", ultralytics.__version__)
    print("ultralytics source:", ultralytics.__file__)
    print("pycocotools:", version("pycocotools"))

    model = YOLO(str(WEIGHTS))
    ultralytics_metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=640,
        batch=16,
        device="0",
        workers=8,
        save_json=True,
        plots=True,
        project=str(OUTPUT_PROJECT),
        name=RUN_NAME,
        exist_ok=False,
        max_det=300,
        iou=0.7,
    )
    save_dir = Path(ultralytics_metrics.save_dir)
    if save_dir != RUN_DIR:
        raise RuntimeError(f"Unexpected evaluation directory: {save_dir}")

    raw_predictions_path = save_dir / "predictions.json"
    ground_truth_path = save_dir / "test_ground_truth_coco.json"
    normalized_predictions_path = save_dir / "predictions_coco.json"
    filename_to_id, image_count, annotation_count = build_coco_ground_truth(ground_truth_path)
    prediction_count = normalize_predictions(raw_predictions_path, normalized_predictions_path, filename_to_id)
    coco_metrics = run_coco_eval(ground_truth_path, normalized_predictions_path)

    summary = {
        "weights": str(WEIGHTS),
        "weights_sha256": sha256(WEIGHTS),
        "data": str(DATA_YAML),
        "split": "test",
        "images": image_count,
        "annotations": annotation_count,
        "predictions": prediction_count,
        "ultralytics_version": ultralytics.__version__,
        "pycocotools_version": version("pycocotools"),
        "ground_truth_sha256": sha256(ground_truth_path),
        "raw_predictions_sha256": sha256(raw_predictions_path),
        "normalized_predictions_sha256": sha256(normalized_predictions_path),
        "ultralytics_metrics": {
            "precision": float(ultralytics_metrics.box.mp),
            "recall": float(ultralytics_metrics.box.mr),
            "mAP50": float(ultralytics_metrics.box.map50),
            "mAP50_95": float(ultralytics_metrics.box.map),
        },
        "coco_eval": coco_metrics,
    }
    summary_path = save_dir / "cocoeval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n========== COCOeval summary JSON ==========")
    print(json.dumps(summary, indent=2))
    print("summary:", summary_path)


if __name__ == "__main__":
    main()
