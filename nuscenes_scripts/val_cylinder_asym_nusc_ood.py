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
from utils.metric_util import fast_hist_crop, per_class_iu
from utils.nuscenes_ood import (
    get_nuscenes_ood_metadata,
    restore_known_predictions,
)

warnings.filterwarnings("ignore")

SUPPORTED_VARIANTS = {"doss", "fr", "ugfa", "fr_ugfa"}


def sample_identifier(index):
    if torch.is_tensor(index):
        index = index.item()
    if isinstance(index, np.generic):
        index = index.item()
    return f"{int(index):06d}"


def main(args):
    device = torch.device("cuda", 0)
    configs = load_config_data(args.config_path)
    dataset_config = configs["dataset_params"]
    train_config = configs["train_data_loader"]
    val_config = configs["val_data_loader"]
    model_config = configs["model_params"]
    train_params = configs["train_params"]

    model_variant = str(model_config.get("model_variant", "")).lower()
    if model_variant not in SUPPORTED_VARIANTS:
        raise ValueError(
            f"This script requires model_params.model_variant in "
            f"{sorted(SUPPORTED_VARIANTS)}, got '{model_variant}'."
        )

    metadata = get_nuscenes_ood_metadata(dataset_config["label_mapping"])
    configured_classes = model_config["num_class"]
    model_config["num_class"] = metadata["num_known_classes"]
    if configured_classes != model_config["num_class"]:
        print(
            f"Adjusting num_class from {configured_classes} to "
            f"{model_config['num_class']} from ood_inv_map."
        )

    prediction_dir = os.path.join(
        args.save_folder, "CSS_results/sequences/08/predictions"
    )
    anomaly_dir = os.path.join(
        args.save_folder, "AnomalyDetection_results/sequences/08/predictions"
    )
    os.makedirs(prediction_dir, exist_ok=True)
    os.makedirs(anomaly_dir, exist_ok=True)

    checkpoint_path = train_params["model_load_path"]
    if not os.path.exists(checkpoint_path):
        latest_path = train_params.get("model_latest_path")
        if latest_path and os.path.exists(latest_path):
            checkpoint_path = latest_path
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = model_builder.build(model_config)
    print(f"Loading checkpoint {checkpoint_path}")
    model = load_checkpoint(checkpoint_path, model)
    model.to(device)
    model.eval()

    _, val_loader = data_builder.build(
        dataset_config,
        train_config,
        val_config,
        grid_size=model_config["output_shape"],
    )
    known_labels = metadata["known_labels"]
    known_label_indices = metadata["known_label_indices"]
    histogram = np.zeros(
        (len(known_labels), len(known_labels)), dtype=np.int64
    )
    progress = tqdm(total=len(val_loader), dynamic_ncols=True)

    with torch.no_grad():
        for _, _, grids, point_labels, point_features, indices in val_loader:
            feature_tensors = [
                torch.as_tensor(features, dtype=torch.float32, device=device)
                for features in point_features
            ]
            grid_tensors = [torch.as_tensor(grid, device=device) for grid in grids]
            _, css_logits, oss_logits = model(
                feature_tensors, grid_tensors, len(feature_tensors)
            )
            predictions = restore_known_predictions(
                torch.argmax(css_logits, dim=1).cpu().numpy(),
                metadata["inverse_values"],
            )
            for sample_index, grid in enumerate(grids):
                point_prediction = predictions[
                    sample_index, grid[:, 0], grid[:, 1], grid[:, 2]
                ]
                histogram += fast_hist_crop(
                    point_prediction,
                    point_labels[sample_index],
                    known_label_indices,
                )
                output_name = sample_identifier(indices[sample_index]) + ".label"
                point_prediction.astype(np.int32).tofile(
                    os.path.join(prediction_dir, output_name)
                )

                point_logits = oss_logits[
                    sample_index,
                    :,
                    grid[:, 0],
                    grid[:, 1],
                    grid[:, 2],
                ].transpose(0, 1)
                anomaly_score = 1.0 - torch.max(point_logits, dim=1).values
                anomaly_score.cpu().numpy().astype(np.float32).tofile(
                    os.path.join(anomaly_dir, output_name)
                )
            progress.update(1)

    progress.close()
    iou = per_class_iu(histogram)
    print("Validation per-class known IoU:")
    for class_name, class_iou in zip(metadata["known_label_names"], iou):
        print(f"{class_name}: {class_iou * 100:.2f}%")
    print(f"Known-class mIoU: {np.nanmean(iou) * 100:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-y", "--config_path", default="../config/nuScenes_ood_final.yaml"
    )
    parser.add_argument("--save_folder", default="../ow3d_data/nuScenes/")
    arguments = parser.parse_args()
    print(" ".join(sys.argv))
    print(arguments)
    main(arguments)
