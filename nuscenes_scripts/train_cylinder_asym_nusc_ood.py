# -*- coding:utf-8 -*-

import argparse
import json
import os
import pickle
import shutil
import sys
import warnings

import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

sys.path.append("..")

from builder import data_builder, model_builder
from config.config import load_config_data
from utils import loss_functions
from utils.load_save_util import load_checkpoint
from utils.metric_util import fast_hist_crop, per_class_iu
from utils.nuscenes_ood import (
    get_nuscenes_ood_metadata,
    mapping_tensor,
    restore_known_predictions,
)

warnings.filterwarnings("ignore")

ARM_VARIANTS = {"arm", "urar"}
DOSS_LOSS_VARIANTS = {"doss", "ugfr"}
SUPPORTED_VARIANTS = ARM_VARIANTS | DOSS_LOSS_VARIANTS


def init_dist():
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group(backend="nccl")
    return rank, local_rank, world_size


def build_distributed_loader(loader, batch_size, rank, world_size, shuffle):
    sampler = DistributedSampler(
        loader.dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
    )
    distributed_loader = DataLoader(
        loader.dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=loader.num_workers,
        collate_fn=loader.collate_fn,
        pin_memory=True,
        drop_last=loader.drop_last,
    )
    return distributed_loader, sampler


def active_voxel_values(dense_tensor, coordinates):
    indices = tuple(coordinates[:, dimension].long() for dimension in range(4))
    return dense_tensor.permute(0, 2, 3, 4, 1)[indices]


def active_voxel_labels(dense_labels, coordinates):
    indices = tuple(coordinates[:, dimension].long() for dimension in range(4))
    return dense_labels[indices]


def compute_losses(
    css_logits,
    oss_logits,
    coordinates,
    raw_voxel_labels,
    ood_map,
    objectosphere_map,
    uses_arm_loss,
    is_train,
    semantic_loss,
    lovasz_loss,
    objectosphere_loss,
    arcface_loss=None,
    center_loss=None,
    contrastive_loss=None,
    mavs=None,
    loss_epoch=0,
):
    css_voxel_labels = ood_map[raw_voxel_labels]
    point_css_labels = active_voxel_labels(css_voxel_labels, coordinates)
    css_point_logits = active_voxel_values(css_logits, coordinates)
    oss_point_logits = active_voxel_values(oss_logits, coordinates)

    loss_semantic = sum(semantic_loss(css_point_logits, point_css_labels))
    loss_semantic += lovasz_loss(
        torch.softmax(css_logits, dim=1), css_voxel_labels - 1, ignore=-1
    )
    objectosphere_labels = active_voxel_labels(
        objectosphere_map[raw_voxel_labels], coordinates
    )
    loss_objectosphere = objectosphere_loss(oss_point_logits, objectosphere_labels)
    semantic_labels = point_css_labels - 1

    if uses_arm_loss:
        loss_arcface = arcface_loss(oss_point_logits, semantic_labels)
        return loss_semantic + 0.3 * loss_objectosphere + 0.5 * loss_arcface

    center_labels = semantic_labels.clone().long()
    center_labels[center_labels < 0] = 66
    loss_center = center_loss(oss_point_logits, center_labels, is_train=is_train)
    loss_contrastive = contrastive_loss(
        mavs, oss_logits, css_voxel_labels, coordinates, loss_epoch
    )
    return (
        loss_semantic
        + 0.5 * loss_objectosphere
        + 0.3 * loss_center
        + 0.5 * loss_contrastive
    )


def synchronize_and_update_centers(center_loss, device, world_size):
    counts = center_loss.count.to(device)
    sums = torch.stack(
        [center_loss.features[index].to(device) for index in range(center_loss.n_classes)]
    ) * counts.unsqueeze(1)
    if world_size > 1:
        dist.all_reduce(sums, op=dist.ReduceOp.SUM)
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    means = sums / counts.clamp_min(1).unsqueeze(1)
    center_loss.features = {
        index: means[index] for index in range(center_loss.n_classes)
    }
    center_loss.count = counts
    center_loss.update()


def current_mavs(center_loss, device):
    if center_loss is None or center_loss.previous_features is None:
        return None
    return torch.stack(
        [
            center_loss.previous_features[index].to(device)
            for index in range(center_loss.n_classes)
        ]
    )


def load_center_state(center_loss, checkpoint_dir, device):
    centers_path = os.path.join(checkpoint_dir, "class_centers.pt")
    if os.path.exists(centers_path):
        state = torch.load(centers_path, map_location="cpu")
        centers = state["centers"].to(device)
        center_loss.previous_features = {
            index: centers[index] for index in range(center_loss.n_classes)
        }
        center_loss.previous_count = state["counts"].to(device)
        return centers_path

    legacy_mavs = os.path.join(checkpoint_dir, "mavs_current.pickle")
    legacy_counts = os.path.join(checkpoint_dir, "previous_count.pickle")
    if os.path.exists(legacy_mavs) and os.path.exists(legacy_counts):
        with open(legacy_mavs, "rb") as file:
            features = pickle.load(file)
        with open(legacy_counts, "rb") as file:
            counts = pickle.load(file)
        if isinstance(features, dict):
            center_loss.previous_features = {
                index: features[index].to(device)
                for index in range(center_loss.n_classes)
            }
        else:
            features = torch.as_tensor(features, device=device)
            center_loss.previous_features = {
                index: features[index] for index in range(center_loss.n_classes)
            }
        center_loss.previous_count = torch.as_tensor(counts, device=device)
        return legacy_mavs
    return None


def save_center_state(center_loss, centers_path, device):
    torch.save(
        {
            "centers": current_mavs(center_loss, device).detach().cpu(),
            "counts": center_loss.previous_count.detach().cpu(),
        },
        centers_path,
    )


def reduce_histogram(histogram, device, world_size):
    tensor = torch.as_tensor(histogram, dtype=torch.long, device=device)
    if world_size > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.cpu().numpy()


def reduce_mean(total, count, device, world_size):
    values = torch.tensor([total, count], dtype=torch.float64, device=device)
    if world_size > 1:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return (values[0] / values[1].clamp_min(1)).item()


def main(args):
    rank, local_rank, world_size = init_dist()
    device = torch.device("cuda", local_rank)
    configs = load_config_data(args.config_path)
    dataset_config = configs["dataset_params"]
    model_config = configs["model_params"]
    train_config = configs["train_data_loader"]
    val_config = configs["val_data_loader"]
    train_params = configs["train_params"]

    model_variant = str(model_config.get("model_variant", "")).lower()
    if model_variant not in SUPPORTED_VARIANTS:
        raise ValueError(
            f"This script requires model_params.model_variant in "
            f"{sorted(SUPPORTED_VARIANTS)}, got '{model_variant}'."
        )
    uses_arm_loss = model_variant in ARM_VARIANTS

    metadata = get_nuscenes_ood_metadata(dataset_config["label_mapping"])
    configured_classes = model_config["num_class"]
    model_config["num_class"] = metadata["num_known_classes"]
    num_classes = model_config["num_class"]
    known_labels = metadata["known_labels"]
    known_label_indices = metadata["known_label_indices"]
    class_names = metadata["known_label_names"]
    ood_map = mapping_tensor(metadata["ood_map"], device)
    objectosphere_map = mapping_tensor(metadata["ood_obj_map"], device)

    latest_path = train_params["model_latest_path"]
    checkpoint_dir = os.path.dirname(latest_path)
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        with open(os.path.join(checkpoint_dir, "args.json"), "w") as file:
            json.dump(vars(args), file, sort_keys=True, indent=2)
        with open(os.path.join(checkpoint_dir, "argsv.txt"), "w") as file:
            file.write(" ".join(sys.argv) + "\n")
        shutil.copy(args.config_path, os.path.join(checkpoint_dir, "config.yaml"))
        loss_path = "ARM ArcFace/Objectosphere" if uses_arm_loss else "DOSS center/contrastive"
        print(
            f"Cylinder3D nuScenes: variant={model_variant}, loss_path={loss_path}, "
            f"world_size={world_size}, per_gpu_batch={train_config['batch_size']}"
        )
        if configured_classes != num_classes:
            print(
                f"Adjusting num_class from {configured_classes} to {num_classes} "
                "from ood_inv_map."
            )

    model = model_builder.build(model_config)
    resume_path = train_params["model_load_path"]
    if not os.path.exists(resume_path) and os.path.exists(latest_path):
        resume_path = latest_path
    if os.path.exists(resume_path):
        if rank == 0:
            print(f"Loading checkpoint {resume_path}")
        model = load_checkpoint(resume_path, model)
    else:
        resume_path = None
    model.to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    optimizer = optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=train_params["learning_rate"],
    )
    train_loader, val_loader = data_builder.build(
        dataset_config,
        train_config,
        val_config,
        grid_size=model_config["output_shape"],
    )
    train_loader, train_sampler = build_distributed_loader(
        train_loader, train_config["batch_size"], rank, world_size, shuffle=True
    )
    val_loader, val_sampler = build_distributed_loader(
        val_loader, val_config["batch_size"], rank, world_size, shuffle=False
    )

    semantic_loss = loss_functions.CrossEntropyLoss(
        weight=np.ones(num_classes), device=device
    )
    lovasz_loss = loss_functions.build_lovasz(
        lovasz=True,
        num_class=num_classes,
        ignore_label=dataset_config["ignore_label"],
    )
    objectosphere_loss = loss_functions.ObjectosphereLoss(sigma=1.0)
    arcface_loss = None
    center_loss = None
    contrastive_loss = None
    centers_path = os.path.join(checkpoint_dir, "class_centers.pt")
    if uses_arm_loss:
        arcface_loss = loss_functions.ArcFace(s=64.0, margin=0.3, ignore_index=-1)
    else:
        center_loss = loss_functions.CenterLoss(n_classes=num_classes)
        contrastive_loss = loss_functions.ContrastiveLoss(n_classes=num_classes)
        loaded_centers = (
            load_center_state(center_loss, checkpoint_dir, device)
            if resume_path is not None
            else None
        )
        if loaded_centers and rank == 0:
            print(f"Loading DOSS class centers {loaded_centers}")

    best_miou = 0.0
    for epoch in range(train_params["max_num_epochs"]):
        model.train()
        train_sampler.set_epoch(epoch)
        local_train_loss = 0.0
        local_train_count = 0
        mavs = current_mavs(center_loss, device)
        progress = tqdm(total=len(train_loader), dynamic_ncols=True) if rank == 0 else None
        for _, voxel_labels, grids, _, point_features in train_loader:
            feature_tensors = [
                torch.as_tensor(features, dtype=torch.float32, device=device)
                for features in point_features
            ]
            grid_tensors = [torch.as_tensor(grid, device=device) for grid in grids]
            voxel_labels = voxel_labels.long().to(device)
            optimizer.zero_grad(set_to_none=True)
            coordinates, css_logits, oss_logits = model(
                feature_tensors, grid_tensors, len(feature_tensors)
            )
            loss = compute_losses(
                css_logits,
                oss_logits,
                coordinates,
                voxel_labels,
                ood_map,
                objectosphere_map,
                uses_arm_loss,
                True,
                semantic_loss,
                lovasz_loss,
                objectosphere_loss,
                arcface_loss,
                center_loss,
                contrastive_loss,
                mavs,
                epoch,
            )
            loss.backward()
            optimizer.step()
            local_train_loss += loss.detach().item()
            local_train_count += 1
            if progress is not None:
                progress.update(1)

        if center_loss is not None:
            synchronize_and_update_centers(center_loss, device, world_size)
            mavs = current_mavs(center_loss, device)
        train_loss = reduce_mean(
            local_train_loss, local_train_count, device, world_size
        )
        if rank == 0:
            progress.close()
            print(f"Epoch {epoch}: train loss {train_loss:.4f}")
            model_to_save = model.module if world_size > 1 else model
            torch.save(model_to_save.state_dict(), latest_path)
            if center_loss is not None:
                save_center_state(center_loss, centers_path, device)
        if world_size > 1:
            dist.barrier()

        model.eval()
        val_sampler.set_epoch(epoch)
        local_histogram = np.zeros(
            (len(known_labels), len(known_labels)), dtype=np.int64
        )
        local_loss = 0.0
        local_count = 0
        with torch.no_grad():
            for _, voxel_labels, grids, point_labels, point_features, _ in val_loader:
                feature_tensors = [
                    torch.as_tensor(features, dtype=torch.float32, device=device)
                    for features in point_features
                ]
                grid_tensors = [torch.as_tensor(grid, device=device) for grid in grids]
                voxel_labels = voxel_labels.long().to(device)
                coordinates, css_logits, oss_logits = model(
                    feature_tensors, grid_tensors, len(feature_tensors)
                )
                loss = compute_losses(
                    css_logits,
                    oss_logits,
                    coordinates,
                    voxel_labels,
                    ood_map,
                    objectosphere_map,
                    uses_arm_loss,
                    False,
                    semantic_loss,
                    lovasz_loss,
                    objectosphere_loss,
                    arcface_loss,
                    center_loss,
                    contrastive_loss,
                    mavs,
                    epoch + 1,
                )
                predictions = restore_known_predictions(
                    torch.argmax(css_logits, dim=1).cpu().numpy(),
                    metadata["inverse_values"],
                )
                for sample_index, grid in enumerate(grids):
                    local_histogram += fast_hist_crop(
                        predictions[
                            sample_index, grid[:, 0], grid[:, 1], grid[:, 2]
                        ],
                        point_labels[sample_index],
                        known_label_indices,
                    )
                local_loss += loss.item()
                local_count += 1

        histogram = reduce_histogram(local_histogram, device, world_size)
        val_loss = reduce_mean(local_loss, local_count, device, world_size)
        if rank == 0:
            iou = per_class_iu(histogram)
            miou = np.nanmean(iou) * 100
            print(
                f"Epoch {epoch}: known-class mIoU {miou:.3f}, "
                f"validation loss {val_loss:.4f}"
            )
            for class_name, class_iou in zip(class_names, iou):
                print(f"{class_name}: {class_iou * 100:.2f}%")
            if miou > best_miou:
                best_miou = miou
                model_to_save = model.module if world_size > 1 else model
                torch.save(model_to_save.state_dict(), train_params["model_save_path"])
        if world_size > 1:
            dist.barrier()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-y", "--config_path", default="../config/nuScenes_ood_final.yaml"
    )
    arguments = parser.parse_args()
    if int(os.environ.get("RANK", 0)) == 0:
        print(" ".join(sys.argv))
        print(arguments)
    main(arguments)
