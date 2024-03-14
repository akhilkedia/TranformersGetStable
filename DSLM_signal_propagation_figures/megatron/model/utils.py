# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Utilities for models."""

import math

import torch

from megatron import get_args

def init_method_normal(sigma):
    """Init method based on N(0, sigma)."""
    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=sigma)

    return init_


def scaled_init_method_normal(sigma, num_layers):
    """Init method based on N(0, sigma/sqrt(2*num_layers)."""
    std = sigma / math.sqrt(2.0 * num_layers)

    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=std)

    return init_

def sigma_attn_v_out(args, layer_number):
    def attn_block(r):
        r_out = (1-p)
        sigma_w1 = math.sqrt(math.sqrt((1-p)/r)/d)
        return sigma_w1, r_out

    def ffn_block(r):
        r_out = (1-p) *  (r + ((1-r**2)**0.5 - r*math.acos(r))/math.pi)
        return r_out
    
    d = args.hidden_size
    p = args.hidden_dropout
    n_layers = args.num_layers
    lamda_sq = 1.0
    beta_sq = 1.0
    if args.moment_control_lambda_beta:
        lamda_sq = args.moment_control_lambda_sq
        beta_sq = args.moment_control_beta_sq
    r0 = 0.221
    sigma_new = 1.0
    sigma_attn_out_v_list = []
    r_in = r0*(1-p)


    r_list = []
    r_list.append(r_in)
    for _ in range(n_layers):
        sigma_attn_out_v, r_out = attn_block(r_in)
        r_in = (lamda_sq * r_in * sigma_new + beta_sq * r_out * 1.0) / (lamda_sq * sigma_new + beta_sq * 1.0)
        sigma_new = lamda_sq * sigma_new + beta_sq * 1.0
        sigma_attn_out_v_list.append(sigma_attn_out_v)
        r_out = ffn_block(r_in)
        r_in = (lamda_sq * r_in * sigma_new + beta_sq * r_out * 1.0) / (lamda_sq * sigma_new + beta_sq * 1.0)
        sigma_new = lamda_sq * sigma_new + beta_sq * 1.0
        r_list.append(r_in)
    # print(r_list, sigma_attn_out_v_list)

    return sigma_attn_out_v_list[layer_number]

def attention_mask_func(attention_scores, attention_mask):
    attention_scores.masked_fill_(attention_mask, -10000.0)
    return attention_scores


def get_linear_layer(rows, columns, init_method):
    """Simple linear layer with weight initialization."""
    layer = torch.nn.Linear(rows, columns)
    if get_args().perform_initialization:
        init_method(layer.weight)
    with torch.no_grad():
        layer.bias.zero_()
    return layer

@torch.jit.script
def gelu_impl(x):
    """OpenAI's gelu implementation."""
    return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * x *
                                       (1.0 + 0.044715 * x * x)))
def openai_gelu(x):
    return gelu_impl(x)

#This is actually Python equivalent of torch.nn.functional.gelu(), also with type hints for ONNX exporter
@torch.jit.script
def erf_gelu(x):
    return x * 0.5 * (torch.erf(x / 1.41421).to(dtype=x.dtype)+torch.ones_like(x).to(dtype=x.dtype))
