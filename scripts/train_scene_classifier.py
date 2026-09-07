from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import random
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from satquery.models.scene import extract_features


EXTENSIONS = {".jpg", ".jpeg", ".png"}


def collect(root: Path) -> list[tuple[Path, str]]:
    samples = sorted(
        (path, path.parent.name.lower())
        for split in ("training_set", "test_set")
        for path in (root / split).rglob("*")
        if path.is_file() and path.suffix.lower() in EXTENSIONS
    )
    if not samples:
        raise ValueError(f"No supported images found under {root}")
    return samples


def deduplicate(samples: list[tuple[Path, str]]) -> tuple[list[tuple[Path, str]], int]:
    seen: set[str] = set()
    unique: list[tuple[Path, str]] = []
    duplicates = 0
    for path, label in samples:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        key = f"{label}:{digest}"
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        unique.append((path, label))
    return unique, duplicates


def stratified_half_split(samples: list[tuple[Path, str]], seed: int) -> tuple[list, list]:
    grouped: dict[str, list[tuple[Path, str]]] = {}
    for sample in samples:
        grouped.setdefault(sample[1], []).append(sample)
    rng = random.Random(seed)
    train, test = [], []
    for label in sorted(grouped):
        group = grouped[label]
        rng.shuffle(group)
        midpoint = len(group) // 2
        train.extend(group[:midpoint])
        test.extend(group[midpoint:])
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def build(samples: list[tuple[Path, str]]) -> tuple[np.ndarray, np.ndarray]:
    features, labels = [], []
    for index, (path, label) in enumerate(samples, 1):
        try:
            with Image.open(path) as image:
                features.append(extract_features(image))
                labels.append(label)
        except Exception as exc:
            print(f"Skipping unreadable image {path}: {exc}")
        if index % 100 == 0:
            print(f"Processed {index}/{len(samples)} images")
    return np.stack(features), np.asarray(labels)


def predict(train_x, train_y, query_x, k=5):
    predictions = []
    for feature in query_x:
        distances = np.mean((train_x - feature) ** 2, axis=1)
        nearest = np.argpartition(distances, k - 1)[:k]
        labels, counts = np.unique(train_y[nearest], return_counts=True)
        predictions.append(labels[np.argmax(counts)])
    return np.asarray(predictions)


def main():
    parser = argparse.ArgumentParser(description="Train the local seven-class scene baseline")
    parser.add_argument("--data", type=Path, default=Path("data/scene_dataset"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/scene_classifier.npz"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--deployment-all",
        action="store_true",
        help="Train the deployable model on every deduplicated supplied image. Use a separate unseen dataset for evaluation.",
    )
    args = parser.parse_args()
    samples, duplicates = deduplicate(collect(args.data))
    print(f"Unique images: {len(samples)} | Removed exact duplicates: {duplicates}")
    if args.deployment_all:
        train_samples, test_samples = samples, []
        print(f"Deployment training -> Training images: {len(train_samples)} | Internal test images: 0")
        print("Evaluation must use a separate unseen labelled dataset.")
    else:
        train_samples, test_samples = stratified_half_split(samples, args.seed)
        print(f"50/50 split -> Training images: {len(train_samples)} | Test images: {len(test_samples)}")
    train_x, train_y = build(train_samples)
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std < 1e-6] = 1.0
    train_normalized = (train_x - mean) / std
    accuracy = np.nan
    if test_samples:
        test_x, test_y = build(test_samples)
        test_normalized = (test_x - mean) / std
        predictions = predict(train_normalized, train_y, test_normalized)
        accuracy = float((predictions == test_y).mean())
        print(f"Test accuracy: {accuracy:.3f} ({int((predictions == test_y).sum())}/{len(test_y)})")
        for label in sorted(np.unique(test_y)):
            selected = test_y == label
            class_accuracy = float((predictions[selected] == test_y[selected]).mean())
            print(f"  {label:10s}: {class_accuracy:.3f} ({selected.sum()} images)")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        features=train_normalized,
        labels=train_y,
        mean=mean,
        std=std,
        test_accuracy=accuracy,
        training_images=len(train_y),
        training_protocol="all_deduplicated_external_test" if args.deployment_all else "deduplicated_stratified_50_50",
        classes=np.unique(train_y),
    )
    print(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()
