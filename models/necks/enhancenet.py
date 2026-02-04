from collections import OrderedDict
from typing import List

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from models.bricks.basic import SpatialAttention
from models.bricks.misc import Conv2dNormActivation


class EnhanceBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation_layer: nn.Module = nn.ReLU,
        inplace: bool = True,
        groups: int = 4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation_layer(inplace=True)
        
        self.conv1 = Conv2dNormActivation(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=groups,
            activation_layer=None,
            inplace=inplace,
        )
        self.conv2 = Conv2dNormActivation(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=groups,
            activation_layer=None,
            inplace=inplace,
        )
        
        self.att_module = SpatialAttention()
        
        if self.in_channels != self.out_channels:
            self.identity = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
            )
        else:
            self.identity = nn.Identity()
    
    def forward(self, x: Tensor, mask: Tensor):
        y = self.conv1(x) + self.conv2(x)
        y = self.att_module(self.activation(y), mask)
        return y + self.identity(x)


class EnhanceLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 3,
        expansion: float = 1.0,
        groups: int = 4,
        norm_layer: nn.Module = nn.BatchNorm2d,
        activation_layer: nn.Module = nn.SiLU,
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = Conv2dNormActivation(
            in_channels,
            hidden_channels,
            kernel_size=1,
            stride=1,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
            inplace=True,
        )
        self.bottlenecks = nn.Sequential(
            *[
                EnhanceBlock(
                    hidden_channels,
                    hidden_channels,
                    groups=groups,
                    activation_layer=activation_layer,
                ) for _ in range(num_blocks)
            ]
        )
        if hidden_channels != out_channels:
            self.conv2 = Conv2dNormActivation(
                hidden_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
            )
        else:
            self.conv2 = nn.Identity()
    
    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        x = self.conv1(x)
        for bottleneck in self.bottlenecks:
            x = bottleneck(x, mask)
        x = self.conv2(x)
        return x


class EnhanceNetwork(nn.Module):
    def __init__(
        self,
        in_channels_list: List[int],
        out_channels_list: List[int],
        groups: int = 4,
        norm_layer: nn.Module = nn.BatchNorm2d,
        activation: nn.Module = nn.SiLU,
        extra_block: bool = False,
    ):
        super(EnhanceNetwork, self).__init__()
        for idx in range(len(in_channels_list)):
            if in_channels_list[idx] == 0:
                raise ValueError("in_channels=0 is currently not supported")
        
        self.lateral_convs = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        for idx in range(1, len(out_channels_list)):
            lateral_conv_module = Conv2dNormActivation(
                out_channels_list[idx],
                out_channels_list[idx - 1],
                kernel_size=1,
                stride=1,
                norm_layer=norm_layer,
                activation_layer=activation,
                inplace=True,
            )
            layer_block_module = EnhanceLayer(
                out_channels_list[idx - 1] * 2,
                out_channels_list[idx - 1],
                groups=groups,
                norm_layer=norm_layer,
                activation_layer=activation,
            )
            self.lateral_convs.append(lateral_conv_module)
            self.layer_blocks.append(layer_block_module)

        # self.downsample_blocks = nn.ModuleList()
        # self.pan_blocks = nn.ModuleList()
        # for idx in range(len(in_channels_list) - 1):
        #     downsample_block_module = Conv2dNormActivation(
        #         out_channels_list[idx],
        #         out_channels_list[idx + 1],
        #         kernel_size=3,
        #         stride=2,
        #         padding=1,
        #         norm_layer=norm_layer,
        #         activation_layer=activation,
        #         inplace=True,
        #     )
        #     pan_block_module = EnhanceLayer(
        #         out_channels_list[idx + 1] * 2,
        #         out_channels_list[idx + 1],
        #         groups=groups,
        #         norm_layer=norm_layer,
        #         activation_layer=activation,
        #     )
        #     self.downsample_blocks.append(downsample_block_module)
        #     self.pan_blocks.append(pan_block_module)
        self.extra_block = extra_block
        
        self.init_weights()
    
    def init_weights(self):
        # initialize parameters now to avoid modifying the initialization of top_blocks
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: OrderedDict, mask: OrderedDict):
        keys = list(x.keys())
        x = list(x.values())
        mask = list(mask.values())
        assert len(x) == len(self.layer_blocks) + 1
        
        # top down path
        results = x
        inner_outs = [results[-1]]
        for idx in range(len(results) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = results[idx - 1]
            feat_high = self.lateral_convs[idx - 1](feat_high)
            inner_outs[0] = feat_high
            upsample_feat = F.interpolate(
                feat_high,
                size=feat_low.shape[-2:],
                mode="nearest",
            )
            inner_out = self.layer_blocks[idx - 1](torch.cat([upsample_feat, feat_low], dim=1), mask[idx - 1])
            inner_outs.insert(0, inner_out)

        # bottom up path
        # results = [inner_outs[0]]
        # for idx in range(len(inner_outs) - 1):
        #     feat_low = results[-1]
        #     feat_high = inner_outs[idx + 1]
        #     downsample_feat = self.downsample_blocks[idx](feat_low)
        #     out = self.pan_blocks[idx](torch.cat([downsample_feat, feat_high], dim=1), mask[idx + 1])
        #     results.append(out)

        # output layer
        output = OrderedDict()
        for idx in range(len(x)):
            output[keys[idx]] = inner_outs[idx]
        # extra block
        if self.extra_block:
            output["pool"] = F.max_pool2d(list(output.values())[-1], 1, 2, 0)
        
        return output
