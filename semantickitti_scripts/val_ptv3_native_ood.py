# -*- coding:utf-8 -*-

import argparse
import os
import sys
import warnings

import numpy as np
import torch
from tqdm import tqdm

sys.path.append("..")

from builder import data_builder, model_builder
from config.config import load_config_data
from utils.load_save_util import load_checkpoint
from utils.metric_util import per_class_iu
from utils.point_native_utils import (
    known_class_histogram,
    pack_point_batch,
    split_point_predictions,
)
from utils.semantickitti_unknown import get_unknown_label_metadata

warnings.filterwarnings("ignore")

SUPPORTED_POINT_VARIANTS = {"ptv3_native", "randla_native"}
SUPPORTED_POINT_DATASETS = {"ptv3_native_dataset_panop", "point_native_dataset_panop"}


def main(args):
    device = torch.device("cuda:0")
    configs = load_config_data(args.config_path)
    dataset_config = configs["dataset_params"]
    model_config = configs["model_params"]
    train_config = configs["train_data_loader"]
    val_config = configs["val_data_loader"]
    train_params = configs["train_params"]

    if model_config.get("model_variant") not in SUPPORTED_POINT_VARIANTS:
        raise ValueError(
            f"This script requires one of model_params.model_variant={sorted(SUPPORTED_POINT_VARIANTS)}."
        )
    if dataset_config.get("dataset_type") not in SUPPORTED_POINT_DATASETS:
        raise ValueError(
            f"This script requires one of dataset_params.dataset_type={sorted(SUPPORTED_POINT_DATASETS)}."
        )

    unknown_meta = get_unknown_label_metadata(
        dataset_config["label_mapping"],
        dataset_config.get("unknown_label"),
        dataset_config.get("unknown_labels"),
    )
    model_config["num_class"] = unknown_meta["num_known_classes"]
    known_labels = unknown_meta["known_labels"]
    known_class_names = [
        name
        for label, name in zip(unknown_meta["unique_label"] + 1, unknown_meta["unique_label_str"])
        if label in known_labels
    ]
    print(f"Using SemanticKITTI unknown labels: {unknown_meta['unknown_labels_display']}")

    css_folder = os.path.join(args.save_folder, "CSS_results/sequences/08/predictions")
    anomaly_folder = os.path.join(
        args.save_folder, "AnomalyDetection_results/sequences/08/predictions"
    )
    os.makedirs(css_folder, exist_ok=True)
    os.makedirs(anomaly_folder, exist_ok=True)

    model = model_builder.build(model_config)
    if not os.path.exists(train_params["model_load_path"]):
        raise FileNotFoundError(f"Checkpoint not found: {train_params['model_load_path']}")
    model = load_checkpoint(train_params["model_load_path"], model)
    model.to(device).eval()

    _, val_loader = data_builder.build(
        dataset_config,
        train_config,
        val_config,
        grid_size=model_config["output_shape"],
    )
    histogram = np.zeros((len(known_labels), len(known_labels)), dtype=np.int64)
    with torch.no_grad():
        progress = tqdm(total=len(val_loader), dynamic_ncols=True)
        for point_coords, point_features, point_labels, scan_indices in val_loader:
            point_coords, point_features, labels, point_batch, lengths = pack_point_batch(
                point_coords, point_features, point_labels, device
            )
            _, css_token_logits, oss_token_logits, inverse, _ = model(
                point_features,
                point_coords,
                point_batch,
                return_token_maps=True,
            )
            css_logits = css_token_logits[inverse]
            oss_logits = oss_token_logits[inverse]
            histogram += known_class_histogram(css_logits, labels, lengths, known_labels)
            split_predictions = split_point_predictions(
                css_logits, oss_logits, lengths, known_labels
            )

            for scan_index, (css_prediction, oss_point_logits) in zip(scan_indices, split_predictions):
                scan_name = f"{int(scan_index):06d}"
                css_prediction.tofile(os.path.join(css_folder, scan_name + ".label"))

                # Match the existing validation score: low maximum OSS response is anomalous.
                # max_values = np.max(oss_point_logits, axis=1)
                # anomaly_score = np.where(max_values <= 0.4, 1.0, 0.1).astype(np.float32)
                max_values = np.linalg.norm(oss_point_logits, axis=1)
                anomaly_score = np.where(max_values <= 1.0, 1.0, 0.1).astype(np.float32)
                anomaly_score.tofile(os.path.join(anomaly_folder, scan_name + ".label"))
            progress.update(1)
        progress.close()

    iou = per_class_iu(histogram)
    print("Validation known-class IoU:")
    for class_name, class_iou in zip(known_class_names, iou):
        print(f"{class_name}: {class_iou * 100:.2f}%")
    print(f"Current known-class mIoU is {np.nanmean(iou) * 100:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--config_path", default="../config/semantickitti_ood_ptv3.yaml")
    parser.add_argument("--save_folder", default="../exp/semantic_kitti/backbone/ptv3_native")
    arguments = parser.parse_args()
    print(" ".join(sys.argv))
    print(arguments)
    main(arguments)
