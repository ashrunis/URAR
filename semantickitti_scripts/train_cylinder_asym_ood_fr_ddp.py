# -*- coding:utf-8 -*-

import os
import time
import argparse
import sys
sys.path.append('..')
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from utils.metric_util import per_class_iu, fast_hist_crop
from builder import data_builder, model_builder
from config.config import load_config_data
import torch.nn.functional as F
from utils.load_save_util import load_checkpoint
from utils.semantickitti_unknown import (
    build_objectosphere_labels,
    collapse_unknown_label,
    get_unknown_label_metadata,
    restore_known_predictions,
)

from utils import loss_functions

import pickle
import warnings
import json
import shutil

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

def init_dist():
    """Initialize distributed training environment."""
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    if world_size > 1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        print(f"Initializing DDP: Rank {rank}/{world_size}, Local Rank {local_rank}")
    
    return rank, local_rank, world_size

def main(args):
    rank, local_rank, world_size = init_dist()

    pytorch_device = torch.device('cuda', local_rank)

    config_path = args.config_path

    configs = load_config_data(config_path)

    dataset_config = configs['dataset_params']
    train_dataloader_config = configs['train_data_loader']
    val_dataloader_config = configs['val_data_loader']

    per_gpu_train_batch_size = train_dataloader_config['batch_size']
    per_gpu_val_batch_size = val_dataloader_config['batch_size']
    global_train_batch_size = per_gpu_train_batch_size * world_size
    global_val_batch_size = per_gpu_val_batch_size * world_size

    model_config = configs['model_params']
    train_hypers = configs['train_params']

    unknown_label_meta = get_unknown_label_metadata(
        dataset_config["label_mapping"],
        dataset_config.get("unknown_label"),
        dataset_config.get("unknown_labels"),
    )
    inferred_num_class = unknown_label_meta["num_known_classes"]
    if model_config['num_class'] != inferred_num_class:
        if rank == 0:
            print(
                f"Adjusting model num_class from {model_config['num_class']} "
                f"to {inferred_num_class} based on unknown labels "
                f"{unknown_label_meta['unknown_labels']}."
            )
        model_config['num_class'] = inferred_num_class

    grid_size = model_config['output_shape']
    num_class = model_config['num_class']
    ignore_label = dataset_config['ignore_label']

    model_load_path = train_hypers['model_load_path']
    model_save_path = train_hypers['model_save_path']
    model_latest_path = train_hypers['model_latest_path']

    log_save_path = os.path.dirname(model_latest_path)
    if rank == 0:
        os.makedirs(log_save_path, exist_ok=True)
        # save args and config files
        with open(os.path.join(log_save_path, "args.json"), "w") as f:
            json.dump(vars(args), f, sort_keys=True, indent=4)

        with open(os.path.join(log_save_path, "argsv.txt"), "w") as f:
            f.write(" ".join(sys.argv))
            f.write("\n")

        shutil.copy(src=config_path, dst=os.path.join(log_save_path, "config.yaml"))

    unknown_labels = unknown_label_meta["unknown_labels"]
    known_labels = unknown_label_meta["known_labels"]
    unique_label = unknown_label_meta["unique_label"]
    unique_label_str = unknown_label_meta["unique_label_str"]
    if rank == 0:
        print(
            f"Using SemanticKITTI unknown labels: "
            f"{unknown_label_meta['unknown_labels_display']}"
        )
        print(
            f"DDP batch size: per_gpu_train={per_gpu_train_batch_size}, "
            f"global_train={global_train_batch_size}, "
            f"per_gpu_val={per_gpu_val_batch_size}, global_val={global_val_batch_size}"
        )

    # build model
    my_model = model_builder.build(model_config)

    if os.path.exists(model_load_path):
        print(f"Loading pre-trained model {model_load_path}")
        my_model = load_checkpoint(model_load_path, my_model)

    my_model.to(pytorch_device)

    if world_size > 1:
        my_model = DDP(my_model, device_ids=[local_rank], find_unused_parameters=True)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, my_model.parameters()), lr=train_hypers["learning_rate"])

    train_loader_single, val_loader_single = data_builder.build(dataset_config,
                                                                train_dataloader_config,
                                                                val_dataloader_config,
                                                                grid_size=grid_size)
    
    train_dataset = train_loader_single.dataset
    val_dataset = val_loader_single.dataset

    # DDP MOD: 创建 DistributedSamplers
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    # DDP MOD: 获取 collate_fn 和 num_workers
    train_collate_fn = train_loader_single.collate_fn
    val_collate_fn = val_loader_single.collate_fn
    num_workers = train_dataloader_config.get('num_workers', 4)

    # DDP MOD: 创建新的 DDP Dataloaders
    train_dataset_loader = DataLoader(
        train_dataset,
        batch_size=per_gpu_train_batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=train_collate_fn,
        pin_memory=True
    )
    
    val_dataset_loader = DataLoader(
        val_dataset,
        batch_size=per_gpu_val_batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        collate_fn=val_collate_fn,
        pin_memory=True
    )

    # build loss functions
    class_weighting = np.ones(num_class)

    loss_sem_func = loss_functions.CrossEntropyLoss(weight=class_weighting, device=pytorch_device)

    lovasz_softmax = loss_functions.build_lovasz(lovasz=True, num_class=num_class, ignore_label=ignore_label)

    loss_objectosphere_func = loss_functions.ObjectosphereLoss(sigma=1.0)

    loss_arcface_func = loss_functions.ArcFace(s=64.0, margin=0.3, ignore_index=-1)

    # training
    epoch = 0
    best_val_miou = 0
    my_model.train()
    global_iter = 0
    val_global_iter = 0
    check_iter = train_hypers['eval_every_n_steps']

    while epoch < train_hypers['max_num_epochs']:
        loss_list = []
        if rank == 0:
            pbar = tqdm(total=len(train_dataset_loader), dynamic_ncols=True)

        for i_iter, (_, train_vox_label, train_grid, _, train_pt_fea) in enumerate(train_dataset_loader):
            train_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in train_pt_fea]
            train_vox_ten = [torch.from_numpy(i).to(pytorch_device) for i in train_grid]
            voxel_label_tensor = train_vox_label.type(torch.LongTensor).to(pytorch_device)

            voxel_label_tensor_obj = voxel_label_tensor.clone()
            voxel_label_tensor = collapse_unknown_label(voxel_label_tensor, unknown_labels)

            # forward + backward + optimize

            # coor_ori: voxel coordinates of each point [pt_num, 4]
            # y_in_normal: output voxel feaures from semantic decoder [B, num_known_class, 480, 360, 32]
            # y_out_normal: output voxel feaures from open-set decoder [B, num_known_class, 480, 360, 32]
            coor_ori, y_in_normal, y_out_arcface = my_model(train_pt_fea_ten, train_vox_ten, per_gpu_train_batch_size)

            pt_label_origin = voxel_label_tensor[coor_ori.permute(1, 0).chunk(chunks=4, dim=0)]

            # point-wise features
            y_in_normal_valid = y_in_normal.permute(0, 2, 3, 4, 1)
            y_in_normal_valid = y_in_normal_valid[coor_ori.permute(1, 0).chunk(chunks=4, dim=0)].squeeze()

            y_out_arcface_valid = y_out_arcface.permute(0, 2, 3, 4, 1)
            y_out_arcface_valid = y_out_arcface_valid[coor_ori.permute(1, 0).chunk(chunks=4, dim=0)].squeeze()

            loss_sem = loss_sem_func(y_in_normal_valid, pt_label_origin.squeeze())
            loss_lovasz = lovasz_softmax(torch.nn.functional.softmax(y_in_normal), (voxel_label_tensor - 1), ignore=-1)
            loss_sem = sum(loss_sem) + loss_lovasz

            semantic_labels = pt_label_origin.squeeze() - 1
            loss_arcface = loss_arcface_func(y_out_arcface_valid, semantic_labels)

            voxel_label_obj = build_objectosphere_labels(
                voxel_label_tensor_obj[coor_ori.permute(1, 0).chunk(chunks=4, dim=0)],
                unknown_labels,
            )
            loss_obj = loss_objectosphere_func(y_out_arcface_valid, voxel_label_obj.squeeze())

            loss = 1.0 * loss_sem + 0.3 * loss_obj + 0.5 * loss_arcface

            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())

            if rank == 0:
                if global_iter % 500 == 0:
                    if len(loss_list) > 0:
                        print('epoch %d iter %5d, loss: %.3f\n' %
                              (epoch, i_iter, np.mean(loss_list)))
                    else:
                        print('loss error')
                
                pbar.update(1)

            optimizer.zero_grad()
            global_iter += 1
            
            # if global_iter % check_iter == 0 and rank == 0:
            #     if len(loss_list) > 0:
            #         print('epoch %d iter %5d, loss: %.3f\n' %
            #               (epoch, i_iter, np.mean(loss_list)))
            #     else:
            #         print('loss error')

        if rank == 0:
            model_to_save = my_model.module if world_size > 1 else my_model
            torch.save(model_to_save.state_dict(), model_latest_path)
            pbar.close()
        
        if world_size > 1:
            dist.barrier()

        # Evaluation
        if rank == 0:
            print("\n ##########################Evaluating##########################")
        my_model.eval()
        hist_list = []
        val_loss_list = []

        val_sampler.set_epoch(epoch)
        
        with torch.no_grad():
            if rank == 0:
                pbar_val = tqdm(total=len(val_dataset_loader), dynamic_ncols=True)
            else:
                pbar_val = None

            for i_iter_val, (_, val_vox_label, val_grid, val_pt_labs, val_pt_fea, idx) in enumerate(
                    val_dataset_loader):

                val_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in val_pt_fea]
                val_grid_ten = [torch.from_numpy(i).to(pytorch_device) for i in val_grid]
                val_label_tensor = val_vox_label.type(torch.LongTensor).to(pytorch_device)

                val_label_tensor_obj = val_label_tensor.clone()
                val_label_tensor = collapse_unknown_label(val_label_tensor, unknown_labels)
                
                coor_ori, y_in_normal, y_out_arcface = my_model(val_pt_fea_ten, val_grid_ten, per_gpu_val_batch_size)
                
                pt_label_origin = val_label_tensor[coor_ori.permute(1, 0).chunk(chunks=4, dim=0)]

                y_in_normal_valid = y_in_normal.permute(0, 2, 3, 4, 1)
                y_in_normal_valid = y_in_normal_valid[coor_ori.permute(1, 0).chunk(chunks=4, dim=0)].squeeze()

                y_out_arcface_valid = y_out_arcface.permute(0, 2, 3, 4, 1)
                y_out_arcface_valid = y_out_arcface_valid[coor_ori.permute(1, 0).chunk(chunks=4, dim=0)].squeeze()

                val_loss_sem = loss_sem_func(y_in_normal_valid, pt_label_origin.squeeze())
                val_loss_lovasz = lovasz_softmax(torch.nn.functional.softmax(y_in_normal), (val_label_tensor - 1), ignore=-1)
                val_loss_sem = sum(val_loss_sem) + val_loss_lovasz

                semantic_labels = pt_label_origin.squeeze() - 1
                val_loss_arcface = loss_arcface_func(y_out_arcface_valid, semantic_labels)

                val_label_obj = build_objectosphere_labels(
                    val_label_tensor_obj[coor_ori.permute(1, 0).chunk(chunks=4, dim=0)],
                    unknown_labels,
                )
                val_loss_obj = loss_objectosphere_func(y_out_arcface_valid, val_label_obj.squeeze())

                loss_val = 1.0 * val_loss_sem + 0.3 * val_loss_obj + 0.5 * val_loss_arcface

                y_in_normal = torch.argmax(y_in_normal, dim=1)
                y_in_normal = y_in_normal.cpu().detach().numpy()

                y_in_normal = restore_known_predictions(y_in_normal, known_labels)

                for count, i_val_grid in enumerate(val_grid):
                    hist_list.append(fast_hist_crop(y_in_normal[
                                                        count, val_grid[count][:, 0], val_grid[count][:, 1],
                                                        val_grid[count][:, 2]], val_pt_labs[count],
                                                    unique_label))
                val_loss_list.append(loss_val.detach().cpu().numpy())

                if rank == 0:
                    pbar_val.update(1)
                val_global_iter += 1
            
            if world_size > 1:
                local_hist_sum = sum(hist_list)
                local_hist_sum_tensor = torch.tensor(local_hist_sum).to(pytorch_device).contiguous()
                dist.all_reduce(local_hist_sum_tensor, op=dist.ReduceOp.SUM)
                global_hist_sum = local_hist_sum_tensor.cpu().numpy()

                local_val_loss = np.sum(val_loss_list)
                local_val_samples = len(val_loss_list)
                loss_data = torch.tensor([local_val_loss, local_val_samples]).to(pytorch_device)
                dist.all_reduce(loss_data, op=dist.ReduceOp.SUM)
                global_avg_loss = (loss_data[0] / loss_data[1]).item()
                
            else:
                global_hist_sum = sum(hist_list)
                global_avg_loss = np.mean(val_loss_list)

        my_model.train()

        if rank == 0:
            iou = per_class_iu(global_hist_sum)
            val_miou = np.nanmean(iou) * 100
            print('Current val miou is %.3f while the best val miou is %.3f' %
                  (val_miou, best_val_miou))

            print('Validation per class iou: ')
            for class_name, class_iou in zip(unique_label_str, iou):
                print('%s : %.2f%%' % (class_name, class_iou * 100))
            
            if pbar_val:
                pbar_val.close()
            del val_vox_label, val_grid, val_pt_fea, val_grid_ten

            if best_val_miou < val_miou:
                print("Better IoU, saving model...")
                best_val_miou = val_miou
                model_to_save = my_model.module if world_size > 1 else my_model
                torch.save(model_to_save.state_dict(), model_save_path)

            print('Current val loss is %.3f' % global_avg_loss)
            
        epoch += 1
        
        if world_size > 1:
            dist.barrier()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-y', '--config_path', default='../config/semantickitti_ood_final.yaml')
    args = parser.parse_args()

    if int(os.environ.get('RANK', 0)) == 0:
        print(' '.join(sys.argv))
        print(args)
    main(args)
