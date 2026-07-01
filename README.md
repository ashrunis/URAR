# 基本介绍
这是我们开放集点云分割工作（Uncertainty Rectified Angular Representation for Open-set Point Cloud Semantic Segmentation）的代码仓库，实现上我们以 DOSS 项目为基础，并进行了修改

## 执行规范
- 所有脚本默认使用名为 doss 的 conda 环境
- 模型文件在 network 目录下，配置文件在 config 目录下，执行脚本在 semantickitti_scripts 或 nuScenes_scripts 目录下
- 可以从 builder/model_builder.py 中修改 Asymm_3d_spconv 的导入来源来切换不同的模型。segmentator_3d_asymm_spconv.py 是 DOSS 的原始模型，对应 semantickitti_scripts 中的 train_cylinder_asym_ood.py 执行脚本；segmentator_3d_asymm_spconv_fr_ugfa.py 是我们修改后的模型对应 semantickitti_scripts中 的 train_cylinder_asym_ood_fr.py 或 train_cylinder_asym_ood_fr_ddp 执行脚本。具体的执行命令可以参考如下

# 复现流程
```
conda activate doss
```
## Training on SemanticKITTI
```
cd semantickitti_scripts
```
```
CUDA_VISIBLE_DEVICES=0 python train_cylinder_asym_ood_fr.py --config_path ../config/semantickitti_ood_final.yaml
```
```
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_cylinder_asym_ood_fr_ddp.py --config_path ../config/semantickitti_ood_final.yaml
```
## Inference on SemanticKITTI
```
cd semantickitti_scripts
```
```
CUDA_VISIBLE_DEVICES=0 python val_cylinder_asym_ood.py --save_folder ~/myproject/DOSS/exp/semantic_kitti/00/ --config_path ../config/semantickitti_ood_final.yaml
```
## Evaluation on SemanticKITTI
```
cd semantic_kitti_api
```
```
python remap_semantic_labels.py --predictions /home/aqy/myproject/DOSS/exp/semantic_kitti/00/CSS_results/ --split valid --inverse
```
```
python evaluate_semantics.py --dataset /home/aqy/dataset/SemanticKITTI/dataset --predictions /home/aqy/myproject/DOSS/exp/semantic_kitti/00/ --split valid
```
## Training on nuScenes
```
cd nuScenes_scripts
```
```
CUDA_VISIBLE_DEVICES=0 python train_cylinder_asym_nuscenes_ood_fr.py
```
```
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_cylinder_asym_nuscenes_ood_fr_ddp.py
```
## Inference on nuScenes
```
cd nuScenes_scripts
```
```
CUDA_VISIBLE_DEVICES=0 python val_cylinder_asym_nusc_ood.py --save_folder ~/myproject/DOSS/exp/nuscenes/00/
```
## Evaluation on nuScenes
```
cd nuScenes_api
```
```
python evaluate_semantics.py --dataset /home/aqy/dataset/nuscenes/ --predictions /home/aqy/myproject/DOSS/exp/nuscenes/00/ --split valid
```