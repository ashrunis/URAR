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
  model_variant: "fr_ugfa"  # doss, fr, ugfa, fr_ugfa
```

Available variants:

```text
doss      network/segmentator_3d_asymm_spconv.py          DOSS baseline
fr        network/segmentator_3d_asymm_spconv_fr.py       Angular/prototype head variant
ugfa      network/segmentator_3d_asymm_spconv_ugfa.py     UGFA-only variant
fr_ugfa   network/segmentator_3d_asymm_spconv_fr_ugfa.py  Full proposed model
```

Keep the selected variant consistent with the training script. Use the DOSS scripts for `model_variant: "doss"` and the FR scripts for `fr`, `ugfa`, or `fr_ugfa`.

## Configuration Notes

Important SemanticKITTI options:

```yaml
model_params:
  model_variant: "fr_ugfa"

dataset_params:
  unknown_label: 5
  # unknown_labels: [6, 7]

train_params:
  model_load_path: "/path/to/checkpoints/semantic_kitti/exp_name/best_model.pt"
  model_save_path: "/path/to/checkpoints/semantic_kitti/exp_name/best_model.pt"
  model_latest_path: "/path/to/checkpoints/semantic_kitti/exp_name/latest_model.pt"
```

`model_load_path` is also used for resume/evaluation. If the file exists, the training script loads it automatically. Use a separate checkpoint directory for every ablation to avoid continuing from an unrelated experiment.

For SemanticKITTI, `unknown_label` is a learning label. The scripts collapse the configured unknown class during training and restore known-class predictions during inference.

## SemanticKITTI

Run commands from `semantickitti_scripts/`.

```bash
cd semantickitti_scripts
```

### Train DOSS Baseline

Set `model_params.model_variant: "doss"`.

```bash
CUDA_VISIBLE_DEVICES=0 python train_cylinder_asym_ood.py \
  --config_path ../config/semantickitti_ood_final.yaml
```

### Train Proposed Model

Set `model_params.model_variant: "fr_ugfa"` for the full model, or use `fr` / `ugfa` for module-level ablations.

Single GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python train_cylinder_asym_ood_fr.py \
  --config_path ../config/semantickitti_ood_final.yaml
```

DDP:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
  train_cylinder_asym_ood_fr_ddp.py \
  --config_path ../config/semantickitti_ood_final.yaml
```

### Inference

`--save_folder` controls where CSS predictions and anomaly scores are written.

```bash
CUDA_VISIBLE_DEVICES=0 python val_cylinder_asym_ood.py \
  --config_path ../config/semantickitti_ood_final.yaml \
  --save_folder /home/aqy/myproject/OWSS/DOSS/exp/semantic_kitti/doss/
```

Output layout:

```text
<save_folder>/
  CSS_results/sequences/08/predictions/*.label
  AnomalyDetection_results/sequences/08/predictions/*.label
```

The anomaly score is computed inside `semantickitti_scripts/val_cylinder_asym_ood.py`. Check this script before score ablations, because the stable branch keeps score variants as code edits rather than config options.

### Evaluation

SemanticKITTI predictions are first inverse-remapped by the official API script, then evaluated.

```bash
cd ../semantic_kitti_api
```

```bash
python remap_semantic_labels.py \
  --predictions /home/aqy/myproject/OWSS/DOSS/exp/semantic_kitti/doss/CSS_results/ \
  --split valid \
  --inverse
```

```bash
python evaluate_semantics.py \
  --dataset /home/aqy/data/SemanticKITTI/dataset \
  --predictions /home/aqy/myproject/OWSS/DOSS/exp/semantic_kitti/doss/ \
  --split valid
```

The evaluator reads `CSS_results` for semantic segmentation and `AnomalyDetection_results` for open-set metrics such as AUPR and AUROC.

## nuScenes

Run commands from `nuScenes_scripts/`.

```bash
cd nuScenes_scripts
```

### Train DOSS Baseline

Set `model_params.model_variant: "doss"`.

```bash
CUDA_VISIBLE_DEVICES=0 python train_cylinder_asym_nuscenes_ood.py \
  --config_path ../config/nuScenes_ood_final.yaml
```

### Train Proposed Model

Set `model_params.model_variant: "fr_ugfa"` for the full model, or use `fr` / `ugfa` for module-level ablations.

Single GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python train_cylinder_asym_nuscenes_ood_fr.py \
  --config_path ../config/nuScenes_ood_final.yaml
```

DDP:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
  train_cylinder_asym_nuscenes_ood_fr_ddp.py \
  --config_path ../config/nuScenes_ood_final.yaml
```

### Inference

```bash
CUDA_VISIBLE_DEVICES=0 python val_cylinder_asym_nusc_ood.py \
  --config_path ../config/nuScenes_ood_final.yaml \
  --save_folder /home/aqy/myproject/DOSS/exp/nuscenes/00/
```

### Evaluation

```bash
cd ../nuScenes_api
```

```bash
python evaluate_semantics.py \
  --dataset /home/aqy/data/nuscenes/ \
  --predictions /home/aqy/myproject/DOSS/exp/nuscenes/00/ \
  --split valid
```

## Experiment Checklist

Before launching a run:

1. Activate the `doss` environment.
2. Check `model_params.model_variant` in the config.
3. Check dataset paths in the config.
4. Use a unique checkpoint directory for the experiment.
5. Use a matching `--save_folder` for inference.
6. For SemanticKITTI, run inverse remapping on `CSS_results` before offline evaluation.

For reviewer ablations, keep the following comparisons isolated by checkpoint path:

```text
DOSS baseline
Angular/prototype head without UGFA
UGFA-only variant
Full FR+UGFA/URAR model
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
