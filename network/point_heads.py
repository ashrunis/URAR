# -*- coding:utf-8 -*-

import torch
from torch import nn
from torch.nn import functional as F


class PrototypeLinearHead(nn.Module):
    """Cosine classifier whose normalized weights act as class prototypes."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.prototypes = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.prototypes)

    def forward(self, features):
        return F.linear(F.normalize(features), F.normalize(self.prototypes))
