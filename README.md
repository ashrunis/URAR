# URAR for Open-set Point Cloud Semantic Segmentation

This repository contains the implementation of **Uncertainty Rectified Angular Representation for Open-set Point Cloud Semantic Segmentation**. The codebase is built on top of the DOSS/REAL Cylinder3D-style open-set segmentation framework and keeps the original experiment workflow used in the stable `main` branch.

The current branch uses a simple script-level workflow: the model variant is selected by `model_params.model_variant` in the config, and each dataset has separate train, validation, and evaluation entry points.

## Repository Layout

```text
builder/                 Model and dataloader builders
config/                  Experiment configs and label mappings
dataloader/              SemanticKITTI and nuScenes dataloaders
network/                 Cylinder3D, DOSS, FR, UGFA/URAR network definitions
semantickitti_scripts/   SemanticKITTI train and inference scripts
nuScenes_scripts/        nuScenes train and inference scripts
semantic_kitti_api/      SemanticKITTI offline evaluation utilities
nuScenes_api/            nuScenes offline evaluation utilities
utils/                   Losses, metrics, checkpoint helpers, unknown-label helpers
```

## Environment

All commands assume the conda environment is named `doss`.

```bash
conda activate doss
```

The code depends on the common Cylinder3D/DOSS stack, including PyTorch, spconv, torch-scatter, torch-cluster, numpy, scipy, scikit-learn, and PyYAML. The exact CUDA/PyTorch/spconv versions should match the local `doss` environment used for the experiments.

## Data

Update dataset paths in:

```text
config/semantickitti_ood_final.yaml
config/nuScenes_ood_final.yaml
```

Expected SemanticKITTI layout:

```text
<SemanticKITTI root>/
  sequences/
    00/
      velodyne/
      labels/
    ...
```

Expected nuScenes layout follows the local lidarseg info files referenced by `config/nuScenes_ood_final.yaml`.

## Model Selection

Select the network with `model_params.model_variant` in the dataset config:

```yaml
model_params:
  model_architecture: "cylinder_asym"
  model_variant: "fr_ugfa"  # doss, fr, ugfa, fr_ugfa, ptv3_native
```

Available variants:

```text
doss      network/segmentator_3d_asymm_spconv.py          DOSS baseline
fr        network/segmentator_3d_asymm_spconv_fr.py       Angular/prototype head variant
ugfa      network/segmentator_3d_asymm_spconv_ugfa.py     UGFA-only variant
fr_ugfa   network/segmentator_3d_asymm_spconv_fr_ugfa.py  Full proposed model
ptv3_native network/ptv3_native.py                         Point-level Cartesian PTv3 with CSS/OSS decoders
```

Keep the selected variant consistent with the training script. Use the Cylinder3D DOSS script for `model_variant: "doss"`, the FR scripts for `fr`, `ugfa`, or `fr_ugfa`, and the dedicated point-based scripts for `ptv3_native`.

## Configuration Notes

Important SemanticKITTI options:

```yaml
model_params:
  model_variant: "fr_ugfa"
  ptv3_patch_size: 128
  ptv3_drop_path: 0.3
  ptv3_enable_flash: False

dataset_params:
  unknown_label: 5
  # unknown_labels: [6, 7]

train_params:
  model_load_path: "/path/to/checkpoints/semantic_kitti/exp_name/best_model.pt"
  model_save_path: "/path/to/checkpoints/semantic_kitti/exp_name/best_model.pt"
  model_latest_path: "/path/to/checkpoints/semantic_kitti/exp_name/latest_model.pt"
```

`ptv3_native` is the point-level alternative. It consumes Cartesian `xyz + remission` directly and performs an independent Cartesian GridSample for each scan. Training losses supervise one sampled representative per occupied cell; validation uses the inverse map to restore predictions for every original point. Its dedicated configuration is `config/semantickitti_ood_ptv3.yaml`.

`model_load_path` is also used for resume/evaluation. If the file exists, the training script loads it automatically. Use a separate checkpoint directory for every ablation to avoid continuing from an unrelated experiment.

For SemanticKITTI, `unknown_label` is a learning label. The scripts collapse the configured unknown class during training and restore known-class predictions during inference.

## SemanticKITTI

Run commands from `semantickitti_scripts/`.

```bash
cd semantickitti_scripts
```

### Train Cylinder3D Variants

The shared trainer selects the model and loss path from `model_params.model_variant`. `doss` and `ugfa` use the DOSS center/contrastive objectives; `fr` and `fr_ugfa` use ArcFace.

Single GPU:

```bash
CUDA_VISIBLE_DEVICES=6 python train_cylinder_asym_ood_ddp.py --config_path ../config/semantickitti_ood_final.yaml
```

DDP:

```bash
CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 --master_port 29500 train_cylinder_asym_ood_ddp.py --config_path ../config/semantickitti_ood_final.yaml
```

### Train Native PTv3

Use `config/semantickitti_ood_ptv3.yaml`, which selects `model_variant: "ptv3_native"` and the native point dataset wrapper.

```bash
CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 --master_port 29502 train_ptv3_native_ood_fr_ddp.py --config_path ../config/semantickitti_ood_ptv3.yaml
```

For single-GPU debugging, run the same script with `CUDA_VISIBLE_DEVICES=0 python` instead of `torchrun`.

### Inference

`--save_folder` controls where CSS predictions and anomaly scores are written.

```bash
python val_cylinder_asym_ood.py --config_path ../config/semantickitti_ood_final.yaml --save_folder ../exp/semantic_kitti/urar/
```

For PTv3:

```bash
python val_ptv3_native_ood.py --config_path ../config/semantickitti_ood_ptv3.yaml --save_folder ../exp/semantic_kitti/ptv3_native/
```
### Evaluation

The repository keeps `semantic_kitti_api/` unchanged as the official reference implementation. It expects CSS predictions in raw SemanticKITTI IDs, so inverse-remap the learning-ID predictions first:

```bash
cd semantic_kitti_api

python remap_semantic_labels.py --predictions ../exp/semantic_kitti/urar/CSS_results/ --split valid --inverse

python evaluate_semantics.py --dataset ~/data/SemanticKITTI/dataset --predictions ../exp/semantic_kitti/urar/ --split valid
```

## nuScenes

Run commands from `nuScenes_scripts/`.

```bash
cd nuScenes_scripts
```

### Train

The shared trainer supports `doss`, `fr`, `ugfa`, and `fr_ugfa`. The selected `model_variant` determines both the network and loss path.

Single GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python train_cylinder_asym_nusc_ood_ddp.py \
  --config_path ../config/nuScenes_ood_final.yaml
```

DDP:

```bash
CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 --master_port 29501 \
  train_cylinder_asym_nusc_ood_ddp.py \
  --config_path ../config/nuScenes_ood_final.yaml
```

`batch_size` is per GPU. Both shared trainers validate after every epoch and save the latest and best checkpoints configured under `train_params`.

### Inference

```bash
python val_cylinder_asym_nusc_ood.py --config_path ../config/nuScenes_ood_final.yaml --save_folder ../exp/nuscenes/00/
```

### Evaluation

```bash
cd nuScenes_api
```

```bash
python evaluate_semantics.py --dataset ~/data/nuscenes/ --predictions ../exp/nuscenes/00/ --split valid
```

## Citation

If this code is useful for your work, please cite the corresponding paper:

```bibtex
@article{urar_open_set_point_cloud_segmentation,
  title = {Uncertainty Rectified Angular Representation for Open-set Point Cloud Semantic Segmentation},
  author = {Anonymous},
  journal = {Under review},
  year = {2026}
}
```
