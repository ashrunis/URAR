# -*- coding:utf-8 -*-

import numpy as np
import spconv.pytorch as spconv
import torch
from torch import nn
from torch.nn import functional as F


def conv3x3(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                             padding=1, bias=False, indice_key=indice_key)


def conv1x3(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(1, 3, 3), stride=stride,
                             padding=(0, 1, 1), bias=False, indice_key=indice_key)


def conv1x1x3(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(1, 1, 3), stride=stride,
                             padding=(0, 0, 1), bias=False, indice_key=indice_key)


def conv1x3x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(1, 3, 1), stride=stride,
                             padding=(0, 1, 0), bias=False, indice_key=indice_key)


def conv3x1x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(3, 1, 1), stride=stride,
                             padding=(1, 0, 0), bias=False, indice_key=indice_key)


def conv3x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(3, 1, 3), stride=stride,
                             padding=(1, 0, 1), bias=False, indice_key=indice_key)


def conv1x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=1, stride=stride,
                             padding=1, bias=False, indice_key=indice_key)


class ResContextBlock(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size=(3, 3, 3), stride=1, indice_key=None):
        super(ResContextBlock, self).__init__()
        self.conv1 = conv1x3(in_filters, out_filters, indice_key=indice_key + "_conv1_bef")
        self.bn0 = nn.BatchNorm1d(out_filters)
        self.act1 = nn.LeakyReLU()

        self.conv1_2 = conv3x1(out_filters, out_filters, indice_key=indice_key + "_conv1_2_bef")
        self.bn0_2 = nn.BatchNorm1d(out_filters)
        self.act1_2 = nn.LeakyReLU()

        self.conv2 = conv3x1(in_filters, out_filters, indice_key=indice_key + "_conv2_bef")
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(out_filters)

        self.conv3 = conv1x3(out_filters, out_filters, indice_key=indice_key + "_conv3_bef")
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm1d(out_filters)

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut = shortcut.replace_feature(self.act1(shortcut.features))
        shortcut = shortcut.replace_feature(self.bn0(shortcut.features))

        shortcut = self.conv1_2(shortcut)
        shortcut = shortcut.replace_feature(self.act1_2(shortcut.features))
        shortcut = shortcut.replace_feature(self.bn0_2(shortcut.features))

        x = self.conv2(x)
        x = x.replace_feature(self.act2(x.features))
        x = x.replace_feature(self.bn1(x.features))

        x = self.conv3(x)
        x = x.replace_feature(self.act3(x.features))
        x = x.replace_feature(self.bn2(x.features))
        x = x.replace_feature(x.features + shortcut.features)

        return x


class ResBlock(nn.Module):
    def __init__(self, in_filters, out_filters, dropout_rate, kernel_size=(3, 3, 3), stride=1,
                 pooling=True, drop_out=True, height_pooling=False, indice_key=None):
        super(ResBlock, self).__init__()
        self.pooling = pooling
        self.drop_out = drop_out

        self.conv1 = conv3x1(in_filters, out_filters, indice_key=indice_key + "_conv1_bef")
        self.act1 = nn.LeakyReLU()
        self.bn0 = nn.BatchNorm1d(out_filters)

        self.conv1_2 = conv1x3(out_filters, out_filters, indice_key=indice_key + "_conv_1_2_bef")
        self.act1_2 = nn.LeakyReLU()
        self.bn0_2 = nn.BatchNorm1d(out_filters)

        self.conv2 = conv1x3(in_filters, out_filters, indice_key=indice_key + "_conv2_bef")
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(out_filters)

        self.conv3 = conv3x1(out_filters, out_filters, indice_key=indice_key + "_conv3_bef")
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm1d(out_filters)

        if pooling:
            if height_pooling:
                self.pool = spconv.SparseConv3d(out_filters, out_filters, kernel_size=3, stride=2,
                                                padding=1, indice_key=indice_key, bias=False)
            else:
                self.pool = spconv.SparseConv3d(out_filters, out_filters, kernel_size=3, stride=(2, 2, 1),
                                                padding=1, indice_key=indice_key, bias=False)
        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut = shortcut.replace_feature(self.act1(shortcut.features))
        shortcut = shortcut.replace_feature(self.bn0(shortcut.features))

        shortcut = self.conv1_2(shortcut)
        shortcut = shortcut.replace_feature(self.act1_2(shortcut.features))
        shortcut = shortcut.replace_feature(self.bn0_2(shortcut.features))

        x = self.conv2(x)
        x = x.replace_feature(self.act2(x.features))
        x = x.replace_feature(self.bn1(x.features))

        x = self.conv3(x)
        x = x.replace_feature(self.act3(x.features))
        x = x.replace_feature(self.bn2(x.features))

        x = x.replace_feature(x.features + shortcut.features)

        if self.pooling:
            resB = self.pool(x)
            return resB, x
        else:
            return x


class UpBlock(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size=(3, 3, 3), indice_key=None, up_key=None):
        super(UpBlock, self).__init__()
        # self.drop_out = drop_out
        self.trans_dilao = conv3x3(in_filters, out_filters, indice_key=indice_key + "new_up")
        self.trans_act = nn.LeakyReLU()
        self.trans_bn = nn.BatchNorm1d(out_filters)

        self.conv1 = conv1x3(out_filters, out_filters, indice_key=indice_key + "_conv1_up")
        self.act1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(out_filters)

        self.conv2 = conv3x1(out_filters, out_filters, indice_key=indice_key + "_conv2_up")
        self.act2 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm1d(out_filters)

        self.conv3 = conv3x3(out_filters, out_filters, indice_key=indice_key + "_conv3_up")
        self.act3 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm1d(out_filters)
        # self.dropout3 = nn.Dropout3d(p=dropout_rate)

        self.up_subm = spconv.SparseInverseConv3d(out_filters, out_filters, kernel_size=3, indice_key=up_key,
                                                  bias=False)

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, skip):
        upA = self.trans_dilao(x)
        upA = upA.replace_feature(self.trans_act(upA.features))
        upA = upA.replace_feature(self.trans_bn(upA.features))

        ## upsample
        upA = self.up_subm(upA)

        upA = upA.replace_feature(upA.features + skip.features)

        upE = self.conv1(upA)
        upE = upE.replace_feature(self.act1(upE.features))
        upE = upE.replace_feature(self.bn1(upE.features))

        upE = self.conv2(upE)
        upE = upE.replace_feature(self.act2(upE.features))
        upE = upE.replace_feature(self.bn2(upE.features))

        upE = self.conv3(upE)
        upE = upE.replace_feature(self.act3(upE.features))
        upE = upE.replace_feature(self.bn3(upE.features))

        return upE


class ReconBlock(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size=(3, 3, 3), stride=1, indice_key=None):
        super(ReconBlock, self).__init__()
        self.conv1 = conv3x1x1(in_filters, out_filters, indice_key=indice_key + "_conv1_recon_bef")
        self.bn0 = nn.BatchNorm1d(out_filters)
        self.act1 = nn.Sigmoid()

        self.conv1_2 = conv1x3x1(in_filters, out_filters, indice_key=indice_key + "_conv1_2_recon_bef")
        self.bn0_2 = nn.BatchNorm1d(out_filters)
        self.act1_2 = nn.Sigmoid()

        self.conv1_3 = conv1x1x3(in_filters, out_filters, indice_key=indice_key + "_conv1_3_recon_bef")
        self.bn0_3 = nn.BatchNorm1d(out_filters)
        self.act1_3 = nn.Sigmoid()

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut = shortcut.replace_feature(self.bn0(shortcut.features))
        shortcut = shortcut.replace_feature(self.act1(shortcut.features))

        shortcut2 = self.conv1_2(x)
        shortcut2 = shortcut2.replace_feature(self.bn0_2(shortcut2.features))
        shortcut2 = shortcut2.replace_feature(self.act1_2(shortcut2.features))

        shortcut3 = self.conv1_3(x)
        shortcut3 = shortcut3.replace_feature(self.bn0_3(shortcut3.features))
        shortcut3 = shortcut3.replace_feature(self.act1_3(shortcut3.features))
        shortcut = shortcut.replace_feature(shortcut.features + shortcut2.features + shortcut3.features)
        
        shortcut = shortcut.replace_feature(shortcut.features * x.features)

        return shortcut
    

class UGFAModule(nn.Module):
    """
    Uncertainty-Guided Feature Adaptation (UGFA) Module
    基于仿射变换的特征调制模块 (类似 SPADE/AdaIN)
    """
    def __init__(self, in_channels):
        super(UGFAModule, self).__init__()
        
        # 1. 生成缩放因子 gamma (Scale)
        self.conv_gamma = spconv.SubMConv3d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=True
        )
        # 2. 生成偏移因子 beta (Shift)
        self.conv_beta = spconv.SubMConv3d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=True
        )
        
        # 3. 实例归一化 (Instance Norm)
        # spconv 的 features 是 [N, C]，我们使用 InstanceNorm1d 将 N 视为序列长度进行归一化
        # 这在点云处理中是常用的归一化手段
        self.norm = nn.InstanceNorm1d(in_channels, affine=False)

    def forward(self, f_css, f_oss):
        """
        Args:
            f_css: CSS 分支的特征 (SparseTensor)
            f_oss: OSS 分支的特征 (SparseTensor) - 作为 Condition
        Returns:
            f_out: 调制后的 CSS 特征 (SparseTensor)
        """
        # 1. 从 OSS 特征学习调制参数
        # gamma, beta: [N, C]
        gamma = self.conv_gamma(f_oss).features
        beta = self.conv_beta(f_oss).features

        # 2. 对 CSS 特征进行归一化
        # Input: [N, C] -> Unsqueeze -> [1, C, N] (Batch=1, Channel=C, Length=N)
        css_feat = f_css.features
        # 注意：这里假设所有体素属于同一个 Batch 的大列表，或者 InstanceNorm 作用于每个 Channel
        norm_feat = self.norm(css_feat.unsqueeze(0)).squeeze(0)

        # 3. 执行仿射变换: Norm * (1 + gamma) + beta
        # 几何不确定性通过 gamma 控制特征强度，通过 beta 控制特征基准
        out_feat = norm_feat * (1 + gamma) + beta
        
        # 4. 残差连接 (Residual Connection)
        # 保留原始语义信息，防止梯度消失
        out_feat = out_feat + css_feat
        
        # 5. 返回替换特征后的 SparseTensor
        return f_css.replace_feature(out_feat)


class Decoder(nn.Module):
    def __init__(self, init_size=16, nclasses=18, indice_key=None):
        super(Decoder, self).__init__()
        self.upBlock0 = UpBlock(16 * init_size, 16 * init_size, indice_key=indice_key+"upD0", up_key="down5")
        self.upBlock1 = UpBlock(16 * init_size, 8 * init_size, indice_key=indice_key+"upD1", up_key="down4")
        self.upBlock2 = UpBlock(8 * init_size, 4 * init_size, indice_key=indice_key+"upD2", up_key="down3")
        self.upBlock3 = UpBlock(4 * init_size, 2 * init_size, indice_key=indice_key+"upD3", up_key="down2")

        self.ReconNet = ReconBlock(2 * init_size, 2 * init_size, indice_key=indice_key+"reconD")

        self.logits = spconv.SubMConv3d(4 * init_size, nclasses, indice_key=indice_key+"logitD", kernel_size=3, stride=1, padding=1, bias=True)

    # === [修改] 接口变更为接收 UGFA 模块和 OSS 特征 ===
    def forward(self, x, ugfa_module=None, oss_features=None, return_features=False):
        down1b, down2b, down3b, down4b, down4c = x
        up4e = self.upBlock0(down4c, down4b)
        up3e = self.upBlock1(up4e, down3b)
        up2e = self.upBlock2(up3e, down2b)
        up1e = self.upBlock3(up2e, down1b)

        up0e = self.ReconNet(up1e)

        up0e = up0e.replace_feature(torch.cat((up0e.features, up1e.features), 1))

        # === [核心修改] UGFA 交互逻辑 ===
        # 如果传入了 UGFA 模块和 OSS 特征，则执行特征调制
        if ugfa_module is not None and oss_features is not None:
            # up0e 是 CSS 特征 (f_css)
            # oss_features 是几何特征 (f_oss)
            up0e = ugfa_module(f_css=up0e, f_oss=oss_features)
        # ===============================

        # 如果需要返回特征给 UAFR 模块 (OSS Decoder)
        if return_features:
            return up0e

        logits = self.logits(up0e)
        logits = logits.dense()
        return logits


class Asymm_3d_spconv(nn.Module):
    def __init__(self,
                 output_shape,
                 num_input_features=128,
                 nclasses=20, init_size=16):
        super(Asymm_3d_spconv, self).__init__()
        self.nclasses = nclasses

        sparse_shape = np.array(output_shape)

        self.sparse_shape = sparse_shape

        self.downCntx = ResContextBlock(num_input_features, init_size, indice_key="pre")
        self.resBlock2 = ResBlock(init_size, 2 * init_size, 0.2, height_pooling=True, indice_key="down2")
        self.resBlock3 = ResBlock(2 * init_size, 4 * init_size, 0.2, height_pooling=True, indice_key="down3")
        self.resBlock4 = ResBlock(4 * init_size, 8 * init_size, 0.2, pooling=True, height_pooling=False, indice_key="down4")
        self.resBlock5 = ResBlock(8 * init_size, 16 * init_size, 0.2, pooling=True, height_pooling=False, indice_key="down5")

        self.upBlock0 = UpBlock(16 * init_size, 16 * init_size, indice_key="up0", up_key="down5")
        self.upBlock1 = UpBlock(16 * init_size, 8 * init_size, indice_key="up1", up_key="down4")
        self.upBlock2 = UpBlock(8 * init_size, 4 * init_size, indice_key="up2", up_key="down3")
        self.upBlock3 = UpBlock(4 * init_size, 2 * init_size, indice_key="up3", up_key="down2")

        self.ReconNet = ReconBlock(2 * init_size, 2 * init_size, indice_key="recon")

        # semantic decoder
        self.decoder_cw = Decoder(init_size=init_size, nclasses=nclasses, indice_key="decode_c_up")
        
        # open-set decoder
        self.decoder_ow = Decoder(init_size=init_size, nclasses=nclasses, indice_key="decode_o_up")

        # === [新增] 初始化 UGFA 模块 ===
        # 输入通道数是 4 * init_size (cat 之后的特征维度)
        # 例如 init_size=16, 则 channels=64
        self.ugfa_module = UGFAModule(in_channels=4 * init_size)

    def forward_ow(self, voxel_features, coors, batch_size):
        coors = coors.int()
        coor_ori = coors.type(torch.LongTensor)

        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape, batch_size)
        ret = self.downCntx(ret)
        down1c, down1b = self.resBlock2(ret)
        down2c, down2b = self.resBlock3(down1c)
        down3c, down3b = self.resBlock4(down2c)
        down4c, down4b = self.resBlock5(down3c)

        outs = [down1b, down2b, down3b, down4b, down4c]

        # === [核心修改] 交互式前向传播流程 ===
        
        # 1. 前向传播 OSS Decoder (几何分支)
        # 仅获取特征 (return_features=True)
        feature_oss = self.decoder_ow(outs, return_features=True)
        
        # 2. 前向传播 CSS Decoder (语义分支) 并应用 UGFA
        # 将 UGFA 模块实例和 OSS 特征传入 CSS Decoder
        # 内部会执行: Norm(f_css) * (1 + gamma(f_oss)) + beta(f_oss)
        y_in = self.decoder_cw(outs, ugfa_module=self.ugfa_module, oss_features=feature_oss)
        
        # 3. 计算 OSS 分支的 Logits
        features_oss = self.decoder_ow.logits(feature_oss)
        y_out = features_oss.dense()

        return coor_ori, y_in, y_out

