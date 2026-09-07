from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from satquery.models.scene import extract_features


EXTENSIONS = {".jpg", ".jpeg", ".png"}


def collect(root: Path):
    samples = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
            continue
        label = path.parent.name.lower()
        if label in {"training_set", "test_set", root.name.lower()}:
            continue
        samples.append((path, label))
    if not samples:
        raise ValueError("No labelled images found. Put images in TEST_ROOT/class_name/image.jpg folders.")
    return samples


def main():
    parser = argparse.ArgumentParser(description="Evaluate SatQuery scene classifier on a separate labelled dataset")
    parser.add_argument("--data", type=Path, required=True, help="Root containing one folder per class")
    parser.add_argument("--model", type=Path, default=Path("artifacts/scene_classifier.npz"))
    parser.add_argument("--report", type=Path, default=Path("artifacts/external_test_predictions.csv"))
    args = parser.parse_args()
    model = np.load(args.model, allow_pickle=False)
    known = {str(value) for value in model["classes"]} if "classes" in model else {str(value) for value in np.unique(model["labels"])}
    samples = collect(args.data)
    unknown = sorted({label for _, label in samples} - known)
    if unknown:
        raise ValueError(f"Unknown class folders: {', '.join(unknown)}. Expected: {', '.join(sorted(known))}")

    rows = []
    for index, (path, truth) in enumerate(samples, 1):
        with Image.open(path) as image:
            feature = extract_features(image)
        normalized = (feature - model["mean"]) / model["std"]
        distances = np.mean((model["features"] - normalized) ** 2, axis=1)
        k = min(5, len(distances))
        nearest = np.argpartition(distances, k - 1)[:k]
        labels, counts = np.unique(model["labels"][nearest], return_counts=True)
        predicted = str(labels[np.argmax(counts)])
        rows.append((str(path), truth, predicted, truth == predicted))
        if index % 100 == 0:
            print(f"Evaluated {index}/{len(samples)} images")

    correct = sum(row[3] for row in rows)
    print(f"External test accuracy: {correct / len(rows):.3f} ({correct}/{len(rows)})")
    for label in sorted(known):
        group = [row for row in rows if row[1] == label]
        if group:
            hits = sum(row[3] for row in group)
            print(f"  {label:10s}: {hits / len(group):.3f} ({hits}/{len(group)})")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["path", "true_label", "predicted_label", "correct"])
        writer.writerows(rows)
    print(f"Predictions saved to {args.report}")


if __name__ == "__main__":
    main()
