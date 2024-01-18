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

import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable

import MultiScaleDeformableAttention as MSDA


class MSDeformAttnFunction(Function):
    @staticmethod
    def forward(ctx, value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, im2col_step):
        ctx.im2col_step = im2col_step
        output = MSDA.ms_deform_attn_forward(
            value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, ctx.im2col_step)
        ctx.save_for_backward(value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights = ctx.saved_tensors
        grad_value, grad_sampling_loc, grad_attn_weight = \
            MSDA.ms_deform_attn_backward(
                value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, grad_output, ctx.im2col_step)

        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None


def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    # for debug and test only,
    # need to use cuda version instead
    N_, S_, M_, D_ = value.shape
    
    # [bs, Len_q, 1, n_levels, n_points, 2]
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape
    
    # 把value分割到各个特征层上得到对应的 list value
    # 分为n_level段
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    
    # 由于以下使用了F.grid_sample(), 要求采样位置的坐标是归一化到(-1, -1)代表左上角, (1, 1)代表右下角
    # 因此这里使用2 * sampling_locations - 1将[0, 1]映射到[-1, 1]
    # F.grid_sample() 将图像看为归一化到[-1, 1]之间，中心点为(0, 0) (-1, -1)代表左上角, (1, 1)代表右下角
    sampling_grids = 2 * sampling_locations - 1
    
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        # 首先是得到每个特征层的value list
        # 其次每个bs的每个头
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_*M_, D_, H_, W_)
        
        # N, Lq_, M_, L_, P_, 2  （Lq_ H*W）
        # sampling_grids[:, :, :, lid_] 得到 [N, Lq_, M_, P_, 2]
        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        # 得到每个特征层的采样点 list
        # 每个特征点有n_level * n_points个采样点，这固定的特征层中，每个特征点有n_point个采样点！没问题
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        
        # N_*M_, D_, Lq_, P_   采样算法  根据每个特征层采样点到每个特征层的value进行采样  非采样点用0填充
        sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_,
                                          mode='bilinear', padding_mode='zeros', align_corners=False)
        sampling_value_list.append(sampling_value_l_)
        
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_, M_, 1, Lq_, L_*P_)  其中1是dim_维度
    attention_weights = attention_weights.transpose(1, 2).reshape(N_*M_, 1, Lq_, L_*P_)
    
    
    # 注意力权重 和 采样后的value 进行 weighted sum
    # (torch.stack(sampling_value_list, dim=-2)
    """
    当你对这个列表执行 torch.stack(sampling_value_list, dim=-2)，你在倒数第二个维度(-2)上堆叠这四个张量。
    这意味着将在这个维度上添加一个新的维度，其大小等于堆叠的张量数，即 4。
    假设 sampling_value_list 中的每个元素都有相同的形状 (N*M, D, Lq, P),
    则 torch.stack(sampling_value_list, dim=-2) 后的结果将有维度 (N*M, D, Lq, 4, P)
    """
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(N_, M_*D_, Lq_)
    return output.transpose(1, 2).contiguous()
