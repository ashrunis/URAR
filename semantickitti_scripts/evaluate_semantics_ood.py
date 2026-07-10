#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import argparse
import os
from pathlib import Path
import sys

import numpy as np
import yaml
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import load_config_data
from utils.semantickitti_unknown import get_unknown_label_metadata


def resolve_config_reference(value, config_path):
    path = Path(os.path.expanduser(value))
    if path.is_absolute():
        return path

    candidates = (
        Path.cwd() / path,
        config_path.parent / path,
        PROJECT_ROOT / path,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not resolve config reference: {value}")


def get_sequences_root(dataset_path):
    dataset_path = Path(os.path.expanduser(dataset_path)).resolve()
    if dataset_path.name == "sequences":
        sequences_root = dataset_path
    else:
        sequences_root = dataset_path / "sequences"
    if not sequences_root.is_dir():
        raise FileNotFoundError(f"SemanticKITTI sequences directory not found: {sequences_root}")
    return sequences_root


def collect_files(root, sequences, relative_dir):
    files = {}
    for sequence in sequences:
        sequence_name = f"{int(sequence):02d}"
        directory = root / relative_dir / sequence_name
        if relative_dir == Path("sequences"):
            directory = directory / "labels"
        else:
            directory = directory / "predictions"

        if not directory.is_dir():
            raise FileNotFoundError(f"Required directory not found: {directory}")
        for path in sorted(directory.glob("*.label")):
            key = (sequence_name, path.stem)
            if key in files:
                raise ValueError(f"Duplicate scan key {key} under {directory}")
            files[key] = path
    return files


def require_matching_scans(labels, predictions, scores):
    label_keys = set(labels)
    prediction_keys = set(predictions)
    score_keys = set(scores)
    if label_keys == prediction_keys == score_keys:
        return sorted(label_keys)

    messages = []
    for name, keys in (("predictions", prediction_keys), ("scores", score_keys)):
        missing = sorted(label_keys - keys)
        extra = sorted(keys - label_keys)
        if missing:
            messages.append(f"{name} missing {len(missing)} scans, first: {missing[:3]}")
        if extra:
            messages.append(f"{name} has {len(extra)} extra scans, first: {extra[:3]}")
    raise ValueError("Scan files do not match: " + "; ".join(messages))


def build_learning_lut(mapping):
    lut = np.zeros(1 << 16, dtype=np.int32)
    raw_ids = np.asarray(list(mapping.keys()), dtype=np.int64)
    if np.any(raw_ids < 0) or np.any(raw_ids >= lut.size):
        raise ValueError("SemanticKITTI raw label IDs must fit in 16 bits.")
    lut[raw_ids] = np.asarray(list(mapping.values()), dtype=np.int32)
    return lut


def update_confusion(confusion, prediction, target):
    num_classes = confusion.shape[0]
    if prediction.size == 0:
        return
    invalid = prediction[(prediction < 0) | (prediction >= num_classes)]
    if invalid.size:
        raise ValueError(f"Prediction contains invalid learning IDs: {np.unique(invalid)[:10]}")
    indices = num_classes * prediction.astype(np.int64) + target.astype(np.int64)
    confusion += np.bincount(indices, minlength=num_classes ** 2).reshape(
        num_classes, num_classes
    )


def compute_known_metrics(confusion, known_labels, ignored_labels):
    filtered = confusion.copy()
    filtered[:, ignored_labels] = 0
    true_positive = np.diag(filtered)
    false_positive = filtered.sum(axis=1) - true_positive
    false_negative = filtered.sum(axis=0) - true_positive
    union = true_positive + false_positive + false_negative

    class_iou = np.full(confusion.shape[0], np.nan, dtype=np.float64)
    valid_union = union > 0
    class_iou[valid_union] = true_positive[valid_union] / union[valid_union]
    known_iou = class_iou[np.asarray(known_labels, dtype=np.int64)]
    mean_iou = float(np.nanmean(known_iou))

    known = np.asarray(known_labels, dtype=np.int64)
    denominator = true_positive[known].sum() + false_positive[known].sum()
    accuracy = float(true_positive[known].sum() / max(denominator, 1))
    return mean_iou, class_iou, accuracy


def compute_ood_metrics(binary_labels, scores):
    labels = np.concatenate(binary_labels)
    scores = np.concatenate(scores)
    if not np.all(np.isfinite(scores)):
        raise ValueError("Anomaly scores contain NaN or infinite values.")
    if np.unique(labels).size != 2:
        raise ValueError("OOD evaluation requires both known and unknown points.")

    precision, recall, _ = precision_recall_curve(labels, scores)
    aupr = float(auc(recall, precision))
    false_positive_rate, true_positive_rate, _ = roc_curve(labels, scores)
    auroc = float(auc(false_positive_rate, true_positive_rate))
    return aupr, auroc, labels.size, int(labels.sum())


def main(args):
    config_path = Path(os.path.expanduser(args.config_path)).resolve()
    configs = load_config_data(str(config_path))
    dataset_config = configs["dataset_params"]
    mapping_path = resolve_config_reference(dataset_config["label_mapping"], config_path)

    with mapping_path.open("r") as stream:
        semantic_config = yaml.safe_load(stream)
    unknown_meta = get_unknown_label_metadata(
        str(mapping_path),
        dataset_config.get("unknown_label"),
        dataset_config.get("unknown_labels"),
    )

    dataset_path = args.dataset or configs["val_data_loader"]["data_path"]
    sequences_root = get_sequences_root(dataset_path)
    prediction_root = Path(os.path.expanduser(args.predictions)).resolve()
    split_name = "valid" if args.split == "val" else args.split
    sequences = semantic_config["split"][split_name]

    label_files = collect_files(sequences_root.parent, sequences, Path("sequences"))
    prediction_files = collect_files(
        prediction_root, sequences, Path("CSS_results/sequences")
    )
    score_files = collect_files(
        prediction_root, sequences, Path("AnomalyDetection_results/sequences")
    )
    scan_keys = require_matching_scans(label_files, prediction_files, score_files)

    learning_lut = build_learning_lut(semantic_config["learning_map"])
    num_classes = len(semantic_config["learning_map_inv"])
    known_labels = unknown_meta["known_labels"]
    unknown_labels = unknown_meta["unknown_labels"]
    ignored_labels = sorted({0, *unknown_labels})
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    ood_labels = []
    ood_scores = []

    print(f"Scans: {len(scan_keys)}")
    print(f"Unknown learning labels: {unknown_meta['unknown_labels_display']}")
    print(f"Prediction space: {args.prediction_space}")

    for key in tqdm(scan_keys, desc="Evaluating", dynamic_ncols=True):
        target_raw = np.fromfile(label_files[key], dtype=np.uint32) & 0xFFFF
        prediction = np.fromfile(prediction_files[key], dtype=np.int32) & 0xFFFF
        score = np.fromfile(score_files[key], dtype=np.float32)

        if args.limit is not None:
            target_raw = target_raw[:args.limit]
            prediction = prediction[:args.limit]
            score = score[:args.limit]
        if not (target_raw.size == prediction.size == score.size):
            raise ValueError(
                f"Point-count mismatch for scan {key}: target={target_raw.size}, "
                f"prediction={prediction.size}, score={score.size}"
            )

        target = learning_lut[target_raw]
        if args.prediction_space == "raw":
            prediction = learning_lut[prediction]

        valid = target != 0
        target = target[valid]
        prediction = prediction[valid]
        score = score[valid]
        update_confusion(confusion, prediction, target)
        ood_labels.append(np.isin(target, unknown_labels).astype(np.uint8))
        ood_scores.append(score.astype(np.float32, copy=False))

    mean_iou, class_iou, accuracy = compute_known_metrics(
        confusion, known_labels, ignored_labels
    )
    aupr, auroc, point_count, unknown_count = compute_ood_metrics(
        ood_labels, ood_scores
    )

    print("\nKnown-class IoU:")
    name_map = {
        label: name
        for label, name in zip(
            unknown_meta["unique_label"] + 1,
            unknown_meta["unique_label_str"],
        )
    }
    for label in known_labels:
        print(f"  {label:2d} {name_map[label]}: {class_iou[label] * 100:.2f}%")
    print(f"Known-class mIoU: {mean_iou * 100:.3f}")
    print(f"Known-class accuracy: {accuracy * 100:.3f}")
    print(f"AUPR: {aupr:.6f}")
    print(f"AUROC: {auroc:.6f}")
    print(
        f"OOD points: {unknown_count}/{point_count} "
        f"({100.0 * unknown_count / point_count:.3f}%)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate SemanticKITTI known-class mIoU and OOD metrics."
    )
    parser.add_argument(
        "--config_path",
        default=str(PROJECT_ROOT / "config/semantickitti_ood_final.yaml"),
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="SemanticKITTI dataset root or sequences directory; defaults to val_data_loader.data_path.",
    )
    parser.add_argument(
        "--predictions",
        required=True,
        help="Experiment directory containing CSS_results and AnomalyDetection_results.",
    )
    parser.add_argument("--split", choices=("train", "val", "valid", "test"), default="valid")
    parser.add_argument(
        "--prediction_space",
        choices=("learning", "raw"),
        default="learning",
        help="Use learning for direct val output, or raw for inverse-remapped predictions.",
    )
    parser.add_argument("--limit", type=int, default=None)
    main(parser.parse_args())
