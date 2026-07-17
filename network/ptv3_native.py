# -*- coding:utf-8 -*-

import torch
from torch import nn
from torch.nn import functional as F
import torch_scatter

from network.cylinder_spconv_3d import register_model
from network.ptv3.model import PointTransformerV3


class PrototypeLinearHead(nn.Module):
    """Cosine classifier whose normalized weights act as class prototypes."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.prototypes = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.prototypes)

    def forward(self, features):
        return F.linear(F.normalize(features), F.normalize(self.prototypes))


@register_model
class ptv3_native_asym(nn.Module):
    """Point-level PTv3 with Cartesian GridSample and CSS/OSS decoders."""

    def __init__(self, model_config):
        super().__init__()
        self.name = "ptv3_native_asym"
        self.nclasses = int(model_config["num_class"])
        self.grid_size = float(model_config.get("ptv3_native_grid_size", 0.05))
        if self.grid_size <= 0:
            raise ValueError("ptv3_native_grid_size must be positive.")

        in_channels = int(model_config["fea_dim"])
        patch_size = int(model_config.get("ptv3_patch_size", 128))
        drop_path = float(model_config.get("ptv3_drop_path", 0.3))
        enable_flash = bool(model_config.get("ptv3_enable_flash", False))

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
            cls_mode=True,
        )
        self.css_decoder = self.backbone.build_decoder()
        self.oss_decoder = self.backbone.build_decoder()
        self.feature_dim = 32
        self.css_head = nn.Linear(self.feature_dim, self.nclasses)
        self.oss_head = PrototypeLinearHead(self.feature_dim, self.nclasses)

    def _grid_sample(self, point_features, point_coords, point_batch):
        """Grid-sample each scan independently and select one point per cell."""
        num_scans = int(point_batch.max().item()) + 1
        origins = torch_scatter.scatter_min(
            point_coords,
            point_batch,
            dim=0,
            dim_size=num_scans,
        )[0]
        grid_coord = torch.floor(
            (point_coords - origins[point_batch]) / self.grid_size
        ).int()
        batch_grid = torch.cat((point_batch.int().unsqueeze(1), grid_coord), dim=1)
        token_grid, inverse = torch.unique(batch_grid, dim=0, return_inverse=True)

        num_tokens = token_grid.shape[0]
        if self.training:
            # Match training-mode GridSample: randomly choose one representative
            # point from every occupied Cartesian cell.
            representative = torch_scatter.scatter_max(
                torch.rand(point_coords.shape[0], device=point_coords.device),
                inverse,
                dim=0,
                dim_size=num_tokens,
            )[1]
        else:
            # Keep validation deterministic while retaining the same tokenization.
            point_indices = torch.arange(point_coords.shape[0], device=point_coords.device)
            representative = torch_scatter.scatter_min(
                point_indices,
                inverse,
                dim=0,
                dim_size=num_tokens,
            )[0]

        token_features = point_features[representative]
        token_coords = point_coords[representative]
        token_batch = token_grid[:, 0].long()
        return (
            token_features,
            token_coords,
            token_grid[:, 1:].int(),
            token_batch,
            inverse,
            representative,
        )

    def forward(self, point_features, point_coords, point_batch, return_token_maps=False):
        if point_features.ndim != 2 or point_coords.ndim != 2:
            raise ValueError("PTv3 native inputs must be two-dimensional tensors.")
        if point_features.shape[0] != point_coords.shape[0] or point_batch.shape[0] != point_coords.shape[0]:
            raise ValueError("Point features, coordinates, and batch indices must have the same length.")
        if point_features.shape[1] != self.backbone.embedding.in_channels:
            raise ValueError(
                f"Expected {self.backbone.embedding.in_channels} input features, "
                f"got {point_features.shape[1]}."
            )

        (
            token_features,
            token_coords,
            token_grid,
            token_batch,
            inverse,
            representative,
        ) = self._grid_sample(point_features, point_coords, point_batch)
        encoded_point = self.backbone.forward_encoder(
            {
                "feat": token_features,
                "coord": token_coords,
                "grid_coord": token_grid,
                "batch": token_batch,
            }
        )
        css_point = self.backbone.forward_decoder(encoded_point, self.css_decoder)
        oss_point = self.backbone.forward_decoder(encoded_point, self.oss_decoder)

        css_token_logits = self.css_head(css_point.feat)
        oss_token_logits = self.oss_head(oss_point.feat)
        if return_token_maps:
            return (
                token_batch,
                css_token_logits,
                oss_token_logits,
                inverse,
                representative,
            )

        # Backward-compatible inference output at the original point order.
        return point_batch, css_token_logits[inverse], oss_token_logits[inverse]
