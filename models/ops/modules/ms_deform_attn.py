# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import warnings
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_

from ..functions import MSDeformAttnFunction


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n-1) == 0) and n != 0


class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels  8 16 32 64
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn("You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                          "which is more efficient in our CUDA implementation.")

        # 用于cuda实现
        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        # *2 是因为 offset x, y
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        
        # 本应该是query和Key计算，但在Deformable DETR中直接使用Q线性得到
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        
        # Wm' 将X映射为value
        self.value_proj = nn.Linear(d_model, d_model)
        # Wm  整合多头attention权重
        self.output_proj = nn.Linear(d_model, d_model)

        # 初始化采样点位置
        self._reset_parameters()

    def _reset_parameters(self):
        # 生成初始化的偏置位置 + 注意力权重初始化
        constant_(self.sampling_offsets.weight.data, 0.)
        
        # Initialize Grid for Sampling Offsets:
        
        # (8,) [0, pi/4, pi/2, 3pi/4, pi, 5pi/4, 3pi/2, 7pi/4]
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        
        # [8, 2] 8个头 -> 8个方向 也就是reference point和它周围的8个参考点  九宫格
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        
        # [n_heads, n_levels, n_points, xy] = [8, 4, 4, 2]
        # (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
        """
       [[[[ 1,  0]]],
        [[[ 1,  1]]],
        [[[ 0,  1]]],
        [[[-1,  1]]],
        [[[-1,  0]]],
        [[[-1, -1]]],
        [[[ 0, -1]]],
        [[[ 1, -1]]]]
        """
        # torch.Size([8, 4, 4, 2])
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        
        # 就相当于对于每个reference 框正方形， 逐渐像外框正方形，框n_points个
        # 从图形上看，形成的偏移位置相当于3x3 5x5 7x7 9x9正方形卷积核 去除中心 中心是参考点
        # 要按格子计算，不要按point计算
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
            
        with torch.no_grad():
            # 把初始化的偏移量的偏置bias设置进去  不计算梯度
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
            
        constant_(self.attention_weights.weight.data, 0.)
        
        # 这里与paper描述的有出入，paper中说bias初始化为1/LK, 其中L为特征层数=4, K为每层的采样点数量=4
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None):
        """
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)
        """
        """
        【encoder】
        query: 4个flatten后的特征图+4个flatten后特征图对应的位置编码 = src_flatten + lvl_pos_embed_flatten
               [bs, H/8 * W/8 + H/16 * W/16 + H/32 * W/32 + H/64 * W/64, 256]
        reference_points: 4个flatten后特征图对应的归一化参考点坐标 每个特征点有4个参考点 xy坐标
                          [bs, H/8 * W/8 + H/16 * W/16 + H/32 * W/32 + H/64 * W/64, 4, 2]
        input_flatten: 4个flatten后的特征图=src_flatten  [bs, H/8 * W/8 + H/16 * W/16 + H/32 * W/32 + H/64 * W/64, 256]
        input_spatial_shapes: 4个flatten后特征图的shape [4, 2]
        input_level_start_index: 4个flatten后特征图对应被flatten后的起始索引 [4]  如[0,15100,18900,19850]
        input_padding_mask: 4个flatten后特征图的mask [bs, H/8 * W/8 + H/16 * W/16 + H/32 * W/32 + H/64 * W/64]
        """
        
        """
        i).将输入input_flatten(对于Encoder就是由backbone输出的特征图变换而来,对于Decoder就是Encoder的输出)
        ii). 这里的query是4个flatten后的特征图+4个flatten后特征图对应的位置编码
        iii). 参考点：
        Encoder:
            各个特征层对应的归一化中心坐标，采样点的坐标：预测的坐标偏移得到的
        Decoder: 
            two-stage:是由Encoder预测的top-k proposal boxes函数得到
            one-stage:是由预设的query embedding经过全连接得到
        """
        N, Len_q, _ = query.shape  # bs   query length(每张图片所有特征点的数量)
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        # X经过线性映射得到value
        value = self.value_proj(input_flatten)
        
        # 将特征图mask过的地方（无效地方）的value用0填充
        # 无效的地方是指之前为了将一个batch中的图像设为一样大，padding了很多0，这些位置对应的mask=1
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
            
        # 由于multi-head，所以将dim->(heads, dim/heads)
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        
        # sampling_offsets初始化权重为0，初始化bias为8个方向  
        # 也就是根据query进行线性映射为 head level n_point 2
        # 线性映射的结果+bias
        # [bs,Len_q,256] -> [bs,Len_q,256] -> [bs, Len_q, n_head, n_level, n_point, 2] = [bs, Len_q, 8, 4, 4, 2]
        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        
        # 将attention初始化为1/LK，由于attention_weights初始化都为0，所以经过softmax后会均匀划分
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
        
        # N, Len_q, n_heads, n_levels, n_points, 2
        # sampling_locations 在[0, 1]中间
        if reference_points.shape[-1] == 2:  # one stage
            # [4, 2]  每个(h, w) -> (w, h)
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            # [bs, Len_q, 1, n_point, 1, 2] + [bs, Len_q, n_head, n_level, n_point, 2] / [1, 1, 1, n_point, 1, 2]
            # -> [bs, Len_q, 1, n_levels, n_points, 2]
            # 参考点 + 偏移量/特征层宽高 = 采样点
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4: # two stage  +  iterative bounding box refinement
            # 前两个是xy 后两个是wh
            # 初始化时offset是在 -n_points ~ n_points 范围之间 这里除以self.n_points是相当于把offset归一化到 0~1
            # 然后再乘以宽高的一半 再加上参考点的中心坐标 这就相当于使得最后的采样点坐标总是位于proposal box内
            # 相当于对采样范围进行了约束 减少了搜索空间
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))
        
        # 每个特征点有n_level * n_point个采样点
        
        # 输入：采样点位置、注意力权重、所有点的value
        # 具体过程：根据采样点位置从所有点的value中拿出对应的value，并且和对应的注意力权重进行weighted sum
        # 调用CUDA实现的MSDeformAttnFunction函数  需要编译
        # [bs, Len_q, 256]
        output = MSDeformAttnFunction.apply(
            value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights, self.im2col_step)
        
        # 最后进行公式中的线性运算
        # [bs, Len_q, 256]
        output = self.output_proj(output)
        return output
