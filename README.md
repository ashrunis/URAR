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

The code depends on the common Cylinder3D/DOSS stack, including PyTorch, spconv, torch-scatter, numpy, scipy, scikit-learn, and PyYAML. The exact CUDA/PyTorch/spconv versions should match the local `doss` environment used for the experiments.

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
  model_variant: "fr_ugfa"  # doss, fr, ugfa, fr_ugfa, ptv3, ptv3_doss
```

Available variants:

```text
doss      network/segmentator_3d_asymm_spconv.py          DOSS baseline
fr        network/segmentator_3d_asymm_spconv_fr.py       Angular/prototype head variant
ugfa      network/segmentator_3d_asymm_spconv_ugfa.py     UGFA-only variant
fr_ugfa   network/segmentator_3d_asymm_spconv_fr_ugfa.py  Full proposed model
ptv3      network/ptv3_spconv_3d.py                       PTv3 shared encoder with CSS/OSS decoders
ptv3_doss network/ptv3_spconv_3d.py                       PTv3 shared encoder with DOSS CSS/OSS decoders
```

Keep the selected variant consistent with the training script. Use the DOSS scripts for `model_variant: "doss"` or `ptv3_doss`, and the FR scripts for `fr`, `ugfa`, `fr_ugfa`, or `ptv3`.

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

The PTv3 variants first mean-pool point features in each cylinder voxel and process the resulting unique sparse voxels. They use a shared four-stage encoder with depths `[2, 2, 6, 2]`, followed by separate CSS and OSS decoders, each with depths `[1, 1, 1]`; the output remains compatible with the repository's dense CSS/OSS interface. `ptv3` uses the URAR prototype cosine OSS head, while `ptv3_doss` uses the original DOSS unconstrained logit head and DOSS losses. Set `ptv3_enable_flash: True` to use FlashAttention; this automatically disables the incompatible FP32 attention and softmax upcasts. Keep `ptv3_patch_size: 128` for a like-for-like kernel comparison, or use 1024 to match the official FlashAttention window.

`model_load_path` is also used for resume/evaluation. If the file exists, the training script loads it automatically. Use a separate checkpoint directory for every ablation to avoid continuing from an unrelated experiment.

For SemanticKITTI, `unknown_label` is a learning label. The scripts collapse the configured unknown class during training and restore known-class predictions during inference.

## SemanticKITTI

Run commands from `semantickitti_scripts/`.

```bash
cd semantickitti_scripts
```

### Train DOSS Baseline

Set `model_params.model_variant: "doss"` for Cylinder3D or `"ptv3_doss"` for the PTv3 backbone with the original DOSS training objective.

```bash
CUDA_VISIBLE_DEVICES=6 python train_cylinder_asym_ood.py --config_path ../config/semantickitti_ood_final.yaml
```

### Train Proposed Model

Set `model_params.model_variant: "fr_ugfa"` for the full model, use `fr` / `ugfa` for module-level ablations, or use `ptv3` for the PointTransformerV3 backbone experiment.

Single GPU:

```bash
CUDA_VISIBLE_DEVICES=6 python train_cylinder_asym_ood_fr.py --config_path ../config/semantickitti_ood_final.yaml
```

DDP:

```bash
CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 --master_port 29501 train_cylinder_asym_ood_fr_ddp.py --config_path ../config/semantickitti_ood_final.yaml
```

### Inference

`--save_folder` controls where CSS predictions and anomaly scores are written.

```bash
python val_cylinder_asym_ood.py --config_path ../config/semantickitti_ood_final.yaml --save_folder ../exp/semantic_kitti/backbone/ptv3/
```

### Evaluation

The repository keeps `semantic_kitti_api/` unchanged as the official reference implementation. It expects CSS predictions in raw SemanticKITTI IDs, so inverse-remap the learning-ID predictions first:

```bash
cd semantic_kitti_api

python remap_semantic_labels.py --predictions ../exp/semantic_kitti/ptv3/CSS_results/ --split valid --inverse

python evaluate_semantics.py --dataset ~/data/SemanticKITTI/dataset --predictions ../exp/semantic_kitti/ptv3/ --split valid
```

## nuScenes

Run commands from `nuScenes_scripts/`.

```bash
cd nuScenes_scripts
```

### Train DOSS Baseline

Set `model_params.model_variant: "doss"` for Cylinder3D or `"ptv3_doss"` for the PTv3 backbone with the original DOSS training objective.

```bash
CUDA_VISIBLE_DEVICES=0 python train_cylinder_asym_nusc_ood.py \
  --config_path ../config/nuScenes_ood_final.yaml
```

### Train Proposed Model

Set `model_params.model_variant: "fr_ugfa"` for the full model, use `fr` / `ugfa` for module-level ablations, or use `ptv3` for the PointTransformerV3 backbone experiment.

Single GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python train_cylinder_asym_nusc_ood_fr.py \
  --config_path ../config/nuScenes_ood_final.yaml
```

DDP:

```bash
CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 \
  train_cylinder_asym_nusc_ood_fr_ddp.py \
  --config_path ../config/nuScenes_ood_final.yaml
```

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
