# -*- coding:utf-8 -*-

import torch
from torch import nn
from torch.nn import functional as F
import torch_scatter

from network.cylinder_spconv_3d import register_model
from network.ptv3.model import PointTransformerV3


class PrototypeLinearHead(nn.Module):
    def __init__(self, in_features, out_features):
        super(PrototypeLinearHead, self).__init__()
        self.prototypes = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.prototypes)

    def forward(self, features):
        return F.linear(F.normalize(features), F.normalize(self.prototypes))


@register_model
class ptv3_asym(nn.Module):
    """Voxelized PTv3 with one shared encoder and CSS/OSS decoder branches."""

    def __init__(self, model_config):
        super().__init__()
        self.name = "ptv3_asym"
        self.sparse_shape = tuple(int(v) for v in model_config["output_shape"])
        self.nclasses = int(model_config["num_class"])
        in_channels = int(model_config["fea_dim"])
        patch_size = int(model_config.get("ptv3_patch_size", 128))
        drop_path = float(model_config.get("ptv3_drop_path", 0.3))
        enable_flash = bool(model_config.get("ptv3_enable_flash", False))

        # FlashAttention requires FP16 attention without the FP32 upcasts.
        # The four-stage encoder and three-stage decoder follow the requested
        # [2, 2, 6, 2] / [1, 1, 1] dual-decoder topology.
        self.backbone = PointTransformerV3(
            in_channels=in_channels,
            order=("z", "z-trans", "hilbert", "hilbert-trans"),
            stride=(2, 2, 2),
            enc_depths=(2, 2, 6, 2),
            enc_channels=(32, 64, 128, 256),
            enc_num_head=(2, 4, 8, 16),
            enc_patch_size=(patch_size,) * 4,
            dec_depths=(1, 1, 1),
            dec_channels=(32, 64, 128),
            dec_num_head=(2, 4, 8),
            dec_patch_size=(patch_size,) * 3,
            mlp_ratio=4,
            drop_path=drop_path,
            shuffle_orders=True,
            enable_rpe=False,
            enable_flash=enable_flash,
            upcast_attention=not enable_flash,
            upcast_softmax=not enable_flash,
            # The task-specific module owns two independent decoders below.
            cls_mode=True,
        )
        self.css_decoder = self.backbone.build_decoder()
        self.oss_decoder = self.backbone.build_decoder()
        self.feature_dim = 32
        self.css_head = nn.Linear(self.feature_dim, self.nclasses)
        self.oss_head = PrototypeLinearHead(self.feature_dim, self.nclasses)

    def _pack_points(self, pt_fea, grid_ind):
        device = next(self.parameters()).device
        batch_grid = []
        for batch_idx, grid in enumerate(grid_ind):
            batch_col = torch.full(
                (grid.shape[0], 1),
                batch_idx,
                dtype=grid.dtype,
                device=grid.device,
            )
            batch_grid.append(torch.cat([batch_col, grid], dim=1))

        point_feat = torch.cat(pt_fea, dim=0).float().to(device)
        point_grid = torch.cat(batch_grid, dim=0).long().to(device)
        return point_feat, point_grid

    def _scatter_points_to_dense(self, point_logits, point_grid, batch_size):
        voxel_grid, inverse = torch.unique(point_grid, return_inverse=True, dim=0)
        voxel_logits = torch_scatter.scatter_mean(point_logits, inverse, dim=0)

        dense = point_logits.new_zeros(
            (
                batch_size,
                self.nclasses,
                self.sparse_shape[0],
                self.sparse_shape[1],
                self.sparse_shape[2],
            )
        )
        b = voxel_grid[:, 0].long()
        x = voxel_grid[:, 1].long()
        y = voxel_grid[:, 2].long()
        z = voxel_grid[:, 3].long()
        dense[b, :, x, y, z] = voxel_logits
        return voxel_grid.long(), dense

    @staticmethod
    def _aggregate_voxel_features(point_feat, point_grid):
        """Create the unique sparse voxels required by PTv3's spconv embedding."""
        voxel_grid, inverse = torch.unique(point_grid, return_inverse=True, dim=0)
        voxel_feat = torch_scatter.scatter_mean(point_feat, inverse, dim=0)
        return voxel_feat, voxel_grid

    def forward(self, train_pt_fea_ten, train_vox_ten, batch_size):
        point_feat, point_grid = self._pack_points(train_pt_fea_ten, train_vox_ten)
        voxel_feat, voxel_grid = self._aggregate_voxel_features(point_feat, point_grid)
        encoded_point = self.backbone.forward_encoder(
            {
                "feat": voxel_feat,
                "coord": voxel_grid[:, 1:].float(),
                "grid_coord": voxel_grid[:, 1:].int(),
                "batch": voxel_grid[:, 0].long(),
            }
        )

        css_point = self.backbone.forward_decoder(encoded_point, self.css_decoder)
        oss_point = self.backbone.forward_decoder(encoded_point, self.oss_decoder)
        css_point_logits = self.css_head(css_point.feat)
        oss_point_logits = self.oss_head(oss_point.feat)

        coor_ori, y_in = self._scatter_points_to_dense(
            css_point_logits,
            voxel_grid,
            batch_size,
        )
        _, y_out = self._scatter_points_to_dense(
            oss_point_logits,
            voxel_grid,
            batch_size,
        )
        return coor_ori, y_in, y_out


@register_model
class ptv3_doss_asym(ptv3_asym):
    """Dual-decoder PTv3 with the original DOSS unconstrained OSS logit head."""

    def __init__(self, model_config):
        super().__init__(model_config)
        self.name = "ptv3_doss_asym"
        self.oss_head = nn.Linear(self.feature_dim, self.nclasses)
