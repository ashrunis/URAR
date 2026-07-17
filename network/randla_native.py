# -*- coding:utf-8 -*-

import torch
from torch import nn
from torch.nn import functional as F
from torch_cluster import knn

from network.cylinder_spconv_3d import register_model
from network.point_heads import PrototypeLinearHead


class SharedMLP(nn.Module):
    """Linear-BN-activation block operating on the last tensor dimension."""

    def __init__(self, in_channels, out_channels, activation=True):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        self.activation = nn.LeakyReLU(0.2, inplace=True) if activation else nn.Identity()

    def forward(self, features):
        shape = features.shape
        features = self.linear(features.reshape(-1, shape[-1]))
        features = self.norm(features)
        features = self.activation(features)
        return features.reshape(*shape[:-1], -1)


class AttentivePooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.score = nn.Linear(in_channels, 1, bias=False)
        self.projection = SharedMLP(in_channels, out_channels)

    def forward(self, neighbor_features):
        attention = F.softmax(self.score(neighbor_features), dim=1)
        features = torch.sum(attention * neighbor_features, dim=1)
        return self.projection(features)


class LocalFeatureAggregation(nn.Module):
    """RandLA-Net local spatial encoding followed by attentive pooling."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.spatial_encoding = SharedMLP(10, in_channels)
        self.pooling = AttentivePooling(in_channels * 2, out_channels)

    def forward(self, coords, features, neighbor_indices):
        neighbor_coords = coords[neighbor_indices]
        center_coords = coords.unsqueeze(1).expand_as(neighbor_coords)
        relative_coords = center_coords - neighbor_coords
        distances = torch.linalg.vector_norm(relative_coords, dim=-1, keepdim=True)
        spatial = torch.cat(
            (center_coords, neighbor_coords, relative_coords, distances),
            dim=-1,
        )
        spatial = self.spatial_encoding(spatial)
        neighbor_features = features[neighbor_indices]
        return self.pooling(torch.cat((neighbor_features, spatial), dim=-1))


class DilatedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid_channels = out_channels // 2
        self.pre = SharedMLP(in_channels, mid_channels)
        self.lfa1 = LocalFeatureAggregation(mid_channels, mid_channels)
        self.lfa2 = LocalFeatureAggregation(mid_channels, out_channels)
        self.post = SharedMLP(out_channels, out_channels * 2, activation=False)
        self.shortcut = SharedMLP(in_channels, out_channels * 2, activation=False)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, coords, features, neighbor_indices):
        residual = self.shortcut(features)
        features = self.pre(features)
        features = self.lfa1(coords, features, neighbor_indices)
        features = self.lfa2(coords, features, neighbor_indices)
        return self.activation(self.post(features) + residual)


@register_model
class randla_native_asym(nn.Module):
    """RandLA-Net backbone with independent CSS and OSS decoder branches."""

    def __init__(self, model_config):
        super().__init__()
        self.name = "randla_native_asym"
        self.nclasses = int(model_config["num_class"])
        self.num_neighbors = int(model_config.get("randla_num_neighbors", 16))
        self.sampling_ratios = tuple(
            int(value)
            for value in model_config.get("randla_sub_sampling_ratio", (4, 4, 4, 4))
        )
        d_out = tuple(
            int(value)
            for value in model_config.get("randla_d_out", (16, 64, 128, 256))
        )
        dropout = float(model_config.get("randla_dropout", 0.5))
        if len(d_out) != 4 or len(self.sampling_ratios) != 4:
            raise ValueError("RandLA-Net requires four encoder widths and four sampling ratios.")
        if self.num_neighbors < 1 or any(ratio < 2 for ratio in self.sampling_ratios):
            raise ValueError("RandLA-Net neighbors must be positive and sampling ratios must be >= 2.")

        self.input_projection = SharedMLP(int(model_config["fea_dim"]), 8)
        encoder_channels = [8] + [2 * width for width in d_out]
        self.encoder = nn.ModuleList(
            DilatedResidualBlock(encoder_channels[index], d_out[index])
            for index in range(4)
        )
        self.bottleneck = SharedMLP(encoder_channels[-1], encoder_channels[-1])

        skip_channels = encoder_channels[1:]
        decoder_channels = list(reversed(skip_channels[:-1])) + [skip_channels[0]]
        self.css_decoder = self._build_decoder(skip_channels, decoder_channels)
        self.oss_decoder = self._build_decoder(skip_channels, decoder_channels)

        feature_dim = decoder_channels[-1]
        self.css_head = nn.Sequential(
            SharedMLP(feature_dim, 64),
            SharedMLP(64, 32),
            nn.Dropout(dropout),
            nn.Linear(32, self.nclasses),
        )
        self.oss_head = nn.Sequential(
            SharedMLP(feature_dim, 64),
            SharedMLP(64, 32),
            nn.Dropout(dropout),
            PrototypeLinearHead(32, self.nclasses),
        )

    @staticmethod
    def _build_decoder(skip_channels, decoder_channels):
        modules = nn.ModuleList()
        current_channels = skip_channels[-1]
        for skip_channels_, out_channels in zip(reversed(skip_channels), decoder_channels):
            modules.append(SharedMLP(current_channels + skip_channels_, out_channels))
            current_channels = out_channels
        return modules

    @staticmethod
    def _knn(reference_coords, query_coords, num_neighbors):
        num_neighbors = min(num_neighbors, reference_coords.shape[0])
        assignment = knn(reference_coords, query_coords, num_neighbors)
        return assignment[1].reshape(query_coords.shape[0], num_neighbors)

    def _sample_indices(self, num_points, ratio, device):
        num_sampled = max(num_points // ratio, 1)
        if self.training:
            return torch.randperm(num_points, device=device)[:num_sampled]
        generator = torch.Generator(device=device)
        generator.manual_seed(num_points)
        return torch.randperm(num_points, device=device, generator=generator)[:num_sampled]

    def _forward_scan(self, point_features, point_coords):
        features = self.input_projection(point_features)
        coords = point_coords
        skip_features = []
        skip_coords = []

        for block, ratio in zip(self.encoder, self.sampling_ratios):
            neighbor_indices = self._knn(coords, coords, self.num_neighbors)
            features = block(coords, features, neighbor_indices)
            skip_features.append(features)
            skip_coords.append(coords)
            sampled = self._sample_indices(coords.shape[0], ratio, coords.device)
            coords = coords[sampled]
            features = features[sampled]

        css_features = self.bottleneck(features)
        oss_features = css_features
        coarse_coords = coords
        for level, (css_decoder, oss_decoder) in enumerate(
            zip(self.css_decoder, self.oss_decoder)
        ):
            skip_index = len(skip_coords) - 1 - level
            fine_coords = skip_coords[skip_index]
            nearest = self._knn(coarse_coords, fine_coords, 1).squeeze(1)
            skip = skip_features[skip_index]
            css_features = css_decoder(torch.cat((css_features[nearest], skip), dim=1))
            oss_features = oss_decoder(torch.cat((oss_features[nearest], skip), dim=1))
            coarse_coords = fine_coords

        return self.css_head(css_features), self.oss_head(oss_features)

    def forward(self, point_features, point_coords, point_batch, return_token_maps=False):
        css_logits = []
        oss_logits = []
        for batch_index in range(int(point_batch.max().item()) + 1):
            sample_mask = point_batch == batch_index
            css_sample, oss_sample = self._forward_scan(
                point_features[sample_mask],
                point_coords[sample_mask],
            )
            css_logits.append(css_sample)
            oss_logits.append(oss_sample)

        css_logits = torch.cat(css_logits, dim=0)
        oss_logits = torch.cat(oss_logits, dim=0)
        if return_token_maps:
            identity = torch.arange(point_coords.shape[0], device=point_coords.device)
            return point_batch, css_logits, oss_logits, identity, identity
        return point_batch, css_logits, oss_logits
