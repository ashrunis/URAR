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
    """PointTransformerV3 backbone with DOSS/URAR-compatible dense outputs."""

    def __init__(self, model_config):
        super().__init__()
        self.name = "ptv3_asym"
        self.sparse_shape = tuple(int(v) for v in model_config["output_shape"])
        self.nclasses = int(model_config["num_class"])
        in_channels = int(model_config["fea_dim"])
        patch_size = int(model_config.get("ptv3_patch_size", 128))
        drop_path = float(model_config.get("ptv3_drop_path", 0.3))

        # Official five-stage PTv3 architecture with the non-Flash fallback.
        self.backbone = PointTransformerV3(
            in_channels=in_channels,
            order=("z", "z-trans", "hilbert", "hilbert-trans"),
            stride=(2, 2, 2, 2),
            enc_depths=(2, 2, 2, 6, 2),
            enc_channels=(32, 64, 128, 256, 512),
            enc_num_head=(2, 4, 8, 16, 32),
            enc_patch_size=(patch_size,) * 5,
            dec_depths=(2, 2, 2, 2),
            dec_channels=(64, 64, 128, 256),
            dec_num_head=(4, 4, 8, 16),
            dec_patch_size=(patch_size,) * 4,
            mlp_ratio=4,
            drop_path=drop_path,
            shuffle_orders=True,
            enable_rpe=False,
            enable_flash=False,
            upcast_attention=True,
            upcast_softmax=True,
        )
        feature_dim = 64
        self.css_head = nn.Linear(feature_dim, self.nclasses)
        self.oss_head = PrototypeLinearHead(feature_dim, self.nclasses)

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

    def forward(self, train_pt_fea_ten, train_vox_ten, batch_size):
        point_feat, point_grid = self._pack_points(train_pt_fea_ten, train_vox_ten)
        point = self.backbone(
            {
                "feat": point_feat,
                "coord": point_grid[:, 1:].float(),
                "grid_coord": point_grid[:, 1:].int(),
                "batch": point_grid[:, 0].long(),
            }
        )

        css_point_logits = self.css_head(point.feat)
        oss_point_logits = self.oss_head(point.feat)

        coor_ori, y_in = self._scatter_points_to_dense(
            css_point_logits,
            point_grid,
            batch_size,
        )
        _, y_out = self._scatter_points_to_dense(
            oss_point_logits,
            point_grid,
            batch_size,
        )
        return coor_ori, y_in, y_out
