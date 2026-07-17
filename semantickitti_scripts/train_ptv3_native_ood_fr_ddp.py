# -*- coding:utf-8 -*-

import argparse
import json
import os
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
from utils.metric_util import per_class_iu
from utils.point_native_utils import known_class_histogram, pack_point_batch, urar_point_losses
from utils.semantickitti_unknown import get_unknown_label_metadata

warnings.filterwarnings("ignore")

SUPPORTED_POINT_VARIANTS = {"ptv3_native"}
SUPPORTED_POINT_DATASETS = {"ptv3_native_dataset_panop"}


def init_dist():
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def reduce_histogram(histogram, device, world_size):
    histogram_tensor = torch.as_tensor(histogram, dtype=torch.long, device=device)
    if world_size > 1:
        dist.all_reduce(histogram_tensor, op=dist.ReduceOp.SUM)
    return histogram_tensor.cpu().numpy()


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

    model_variant = model_config.get("model_variant")
    if model_variant not in SUPPORTED_POINT_VARIANTS:
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

    latest_path = train_params["model_latest_path"]
    if rank == 0:
        os.makedirs(os.path.dirname(latest_path), exist_ok=True)
        with open(os.path.join(os.path.dirname(latest_path), "args.json"), "w") as file:
            json.dump(vars(args), file, sort_keys=True, indent=2)
        with open(os.path.join(os.path.dirname(latest_path), "argsv.txt"), "w") as file:
            file.write(" ".join(sys.argv) + "\n")
        shutil.copy(args.config_path, os.path.join(os.path.dirname(latest_path), "config.yaml"))
        print(f"Using SemanticKITTI unknown labels: {unknown_meta['unknown_labels_display']}")
        model_details = (
            f", Cartesian GridSample={model_config['ptv3_native_grid_size']} m"
            if model_variant == "ptv3_native"
            else ""
        )
        print(
            f"Native point DDP: variant={model_variant}, world_size={world_size}, "
            f"per_gpu_batch={train_config['batch_size']}{model_details}"
        )

    model = model_builder.build(model_config)
    if os.path.exists(train_params["model_load_path"]):
        if rank == 0:
            print(f"Loading checkpoint {train_params['model_load_path']}")
        model = load_checkpoint(train_params["model_load_path"], model)
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
    train_sampler = DistributedSampler(
        train_loader.dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    val_sampler = DistributedSampler(
        val_loader.dataset, num_replicas=world_size, rank=rank, shuffle=False
    )
    train_loader = DataLoader(
        train_loader.dataset,
        batch_size=train_config["batch_size"],
        sampler=train_sampler,
        num_workers=train_config["num_workers"],
        collate_fn=train_loader.collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_loader.dataset,
        batch_size=val_config["batch_size"],
        sampler=val_sampler,
        num_workers=val_config["num_workers"],
        collate_fn=val_loader.collate_fn,
        pin_memory=True,
    )

    loss_sem_func = loss_functions.CrossEntropyLoss(
        weight=np.ones(model_config["num_class"]), device=device
    )
    loss_objectosphere_func = loss_functions.ObjectosphereLoss(sigma=1.0)
    loss_arcface_func = loss_functions.ArcFace(s=64.0, margin=0.3, ignore_index=-1)

    best_miou = 0.0
    for epoch in range(train_params["max_num_epochs"]):
        model.train()
        train_sampler.set_epoch(epoch)
        losses = []
        progress = tqdm(total=len(train_loader), dynamic_ncols=True) if rank == 0 else None

        for point_coords, point_features, point_labels in train_loader:
            point_coords, point_features, labels, point_batch, _ = pack_point_batch(
                point_coords, point_features, point_labels, device
            )
            optimizer.zero_grad(set_to_none=True)
            _, css_logits, oss_logits, _, representative = model(
                point_features,
                point_coords,
                point_batch,
                return_token_maps=True,
            )
            token_labels = labels[representative]
            loss, _, _, _ = urar_point_losses(
                css_logits,
                oss_logits,
                token_labels,
                unknown_meta["unknown_labels"],
                loss_sem_func,
                loss_objectosphere_func,
                loss_arcface_func,
            )
            loss.backward()
            optimizer.step()
            losses.append(loss.detach().item())
            if progress is not None:
                progress.update(1)

        if progress is not None:
            progress.close()
            print(f"Epoch {epoch}: train loss {np.mean(losses):.4f}")
            model_to_save = model.module if world_size > 1 else model
            torch.save(model_to_save.state_dict(), train_params["model_latest_path"])
        if world_size > 1:
            dist.barrier()

        model.eval()
        local_histogram = np.zeros((len(known_labels), len(known_labels)), dtype=np.int64)
        local_val_loss = 0.0
        local_val_count = 0
        with torch.no_grad():
            for point_coords, point_features, point_labels, _ in val_loader:
                point_coords, point_features, labels, point_batch, lengths = pack_point_batch(
                    point_coords, point_features, point_labels, device
                )
                _, css_token_logits, oss_token_logits, inverse, representative = model(
                    point_features,
                    point_coords,
                    point_batch,
                    return_token_maps=True,
                )
                token_labels = labels[representative]
                loss, _, _, _ = urar_point_losses(
                    css_token_logits,
                    oss_token_logits,
                    token_labels,
                    unknown_meta["unknown_labels"],
                    loss_sem_func,
                    loss_objectosphere_func,
                    loss_arcface_func,
                )
                # Validation metrics are restored to every original point.
                css_point_logits = css_token_logits[inverse]
                local_histogram += known_class_histogram(
                    css_point_logits,
                    labels,
                    lengths,
                    known_labels,
                )
                local_val_loss += loss.item()
                local_val_count += 1

        histogram = reduce_histogram(local_histogram, device, world_size)
        val_loss = reduce_mean(local_val_loss, local_val_count, device, world_size)
        if rank == 0:
            iou = per_class_iu(histogram)
            miou = np.nanmean(iou) * 100
            print(f"Epoch {epoch}: known-class mIoU {miou:.3f}, validation loss {val_loss:.4f}")
            for class_name, class_iou in zip(known_class_names, iou):
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
    parser.add_argument("-y", "--config_path", default="../config/semantickitti_ood_ptv3.yaml")
    arguments = parser.parse_args()
    if int(os.environ.get("RANK", 0)) == 0:
        print(" ".join(sys.argv))
        print(arguments)
    main(arguments)
