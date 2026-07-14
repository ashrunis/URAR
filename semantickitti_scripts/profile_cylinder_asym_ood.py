# -*- coding:utf-8 -*-
"""Validate SemanticKITTI OOD predictions and profile model inference.

The reported latency and memory cover the model forward pass only.  Host-side
data loading, CPU-to-GPU copies, metric calculation, and prediction writes are
intentionally excluded so the numbers describe inference cost consistently.
"""

import os
import argparse
import sys

sys.path.append('..')

import numpy as np
import torch
from tqdm import tqdm

from utils.metric_util import per_class_iu, fast_hist_crop
from builder import data_builder, model_builder
from config.config import load_config_data
from utils.semantickitti_unknown import (
    get_unknown_label_metadata,
    restore_known_predictions,
)
from utils.load_save_util import load_checkpoint

import warnings

warnings.filterwarnings("ignore")


def count_parameters(model):
    """Return the total number of model parameters in millions."""
    return sum(parameter.numel() for parameter in model.parameters()) / 1e6


def forward_with_profile(model, point_features, grids, batch_size, device):
    """Run one forward pass and return outputs, elapsed time, and peak memory."""
    torch.cuda.reset_peak_memory_stats(device)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    outputs = model(point_features, grids, batch_size)
    end_event.record()
    torch.cuda.synchronize(device)

    latency_ms = start_event.elapsed_time(end_event)
    peak_memory_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    return outputs, latency_ms, peak_memory_gb


def main(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to profile GPU inference metrics.")

    pytorch_device = torch.device('cuda:0')
    config_path = args.config_path
    configs = load_config_data(config_path)

    dataset_config = configs['dataset_params']
    train_dataloader_config = configs['train_data_loader']
    val_dataloader_config = configs['val_data_loader']

    val_batch_size = val_dataloader_config['batch_size']
    model_config = configs['model_params']
    train_hypers = configs['train_params']

    unknown_label_meta = get_unknown_label_metadata(
        dataset_config["label_mapping"],
        dataset_config.get("unknown_label"),
        dataset_config.get("unknown_labels"),
    )
    inferred_num_class = unknown_label_meta["num_known_classes"]
    if model_config['num_class'] != inferred_num_class:
        print(
            f"Adjusting model num_class from {model_config['num_class']} "
            f"to {inferred_num_class} based on unknown labels "
            f"{unknown_label_meta['unknown_labels']}."
        )
        model_config['num_class'] = inferred_num_class

    grid_size = model_config['output_shape']
    ignore_label = dataset_config['ignore_label']
    model_load_path = train_hypers['model_load_path']

    pred_save_folder = os.path.join(args.save_folder, 'CSS_results/sequences/08/predictions/')
    ad_save_folder = os.path.join(
        args.save_folder, 'AnomalyDetection_results/sequences/08/predictions/'
    )
    os.makedirs(pred_save_folder, exist_ok=True)
    os.makedirs(ad_save_folder, exist_ok=True)

    known_labels = unknown_label_meta["known_labels"]
    unique_label = unknown_label_meta["unique_label"]
    unique_label_str = unknown_label_meta["unique_label_str"]
    print(
        f"Using SemanticKITTI unknown labels: "
        f"{unknown_label_meta['unknown_labels_display']}"
    )

    my_model = model_builder.build(model_config)
    try:
        print(f"Loading pre-trained model {model_load_path}")
        my_model = load_checkpoint(model_load_path, my_model)
    except Exception as error:
        print(error)
        print("Error loading pre-trained model.")
        return

    my_model.to(pytorch_device)
    my_model.eval()
    parameter_count_m = count_parameters(my_model)

    _, val_dataset_loader = data_builder.build(
        dataset_config,
        train_dataloader_config,
        val_dataloader_config,
        grid_size=grid_size,
    )

    hist_list = []
    total_profile_latency_ms = 0.0
    profiled_sample_count = 0
    peak_memory_gb = []
    pbar = tqdm(total=len(val_dataset_loader), dynamic_ncols=True)

    with torch.no_grad():
        for i_iter_val, (_, val_vox_label, val_grid, val_pt_labs, val_pt_fea, idx) in enumerate(
            val_dataset_loader
        ):
            val_pt_fea_ten = [
                torch.from_numpy(feature).type(torch.FloatTensor).to(pytorch_device)
                for feature in val_pt_fea
            ]
            val_grid_ten = [torch.from_numpy(grid).to(pytorch_device) for grid in val_grid]
            current_batch_size = len(val_pt_fea_ten)

            if i_iter_val < args.warmup_iters:
                # Warm-up avoids measuring CUDA kernel compilation and allocator start-up.
                coor_ori, y_in_normal, y_out_normal = my_model(
                    val_pt_fea_ten, val_grid_ten, val_batch_size
                )
            else:
                (coor_ori, y_in_normal, y_out_normal), batch_latency_ms, batch_peak_memory_gb = (
                    forward_with_profile(
                        my_model,
                        val_pt_fea_ten,
                        val_grid_ten,
                        val_batch_size,
                        pytorch_device,
                    )
                )
                total_profile_latency_ms += batch_latency_ms
                profiled_sample_count += current_batch_size
                peak_memory_gb.append(batch_peak_memory_gb)

            batch = 0
            idx_s = "%06d" % idx[0]

            y_in_normal = torch.argmax(y_in_normal, dim=1)
            y_in_normal = y_in_normal.cpu().detach().numpy()
            y_in_normal = restore_known_predictions(y_in_normal, known_labels)

            for count, i_val_grid in enumerate(val_grid):
                hist_list.append(
                    fast_hist_crop(
                        y_in_normal[
                            count,
                            val_grid[count][:, 0],
                            val_grid[count][:, 1],
                            val_grid[count][:, 2],
                        ],
                        val_pt_labs[count],
                        unique_label,
                    )
                )

            point_predict = y_in_normal[
                batch,
                val_grid[batch][:, 0],
                val_grid[batch][:, 1],
                val_grid[batch][:, 2],
            ].astype(np.int32)
            point_predict.tofile(os.path.join(pred_save_folder, idx_s + '.label'))

            y_out_normal_pointwise = y_out_normal[
                batch,
                :,
                val_grid[batch][:, 0],
                val_grid[batch][:, 1],
                val_grid[batch][:, 2],
            ].permute(1, 0).cpu().numpy()
            conf_score = (1.0 - np.max(y_out_normal_pointwise, axis=1)).astype(np.float32)
            conf_score.tofile(os.path.join(ad_save_folder, idx_s + ".label"))

            del coor_ori, y_out_normal
            pbar.update(1)

    pbar.close()
    iou = per_class_iu(sum(hist_list))
    val_miou = np.nanmean(iou) * 100
    print('Validation per class iou: ')
    for class_name, class_iou in zip(unique_label_str, iou):
        print('%s : %.2f%%' % (class_name, class_iou * 100))

    print('Current val miou is %.3f' % val_miou)
    if not profiled_sample_count:
        print(
            "No profiling batches were measured. Set --warmup_iters lower than "
            "the validation dataloader length."
        )
        return

    print('\nInference metrics (forward pass only; warm-up batches excluded):')
    print(f"Params (M):  {parameter_count_m:.3f}")
    print(f"Memory (GB): {max(peak_memory_gb):.3f} (peak allocated CUDA memory)")
    print(
        f"Latency (ms): {total_profile_latency_ms / profiled_sample_count:.3f} "
        "(mean per sample)"
    )
    print(f"Profiled samples: {profiled_sample_count}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SemanticKITTI OOD inference profiler')
    parser.add_argument('-y', '--config_path', default='../config/semantickitti_ood_final.yaml')
    parser.add_argument('--save_folder', default='../ow3d_data/SemanticKITTI/')
    parser.add_argument(
        '--warmup_iters',
        type=int,
        default=10,
        help='Number of initial validation batches excluded from profiling.',
    )
    args = parser.parse_args()

    print(' '.join(sys.argv))
    print(args)
    main(args)
