"""Short-window ModernTCN adaptation, independently implemented.

Reference: luodhhh/ModernTCN, ModernTCN-Long-term-forecasting/models/ModernTCN.py
Block's time -> feature -> variable mixing. Unlike the official patch stem,
our required Linear(147,128) is reshaped into 16 learned groups x 8 features.
Groups are latent, NOT original financial variables. See docs/week2/member_c.md.
"""
from dataclasses import dataclass
import math
import torch
from torch import nn


@dataclass(frozen=True)
class QuantModelConfig:
    input_factors: int = 147
    quant_window: int = 12
    d_model: int = 128
    latent_groups: int = 16
    num_blocks: int = 3
    kernel_size: int = 7
    ffn_ratio: int = 2
    dropout: float = 0.1
    head_hidden: int = 64
    pooling: str = 'mean'

    def __post_init__(self):
        for name in ('input_factors', 'quant_window', 'd_model', 'latent_groups',
                     'num_blocks', 'kernel_size', 'ffn_ratio', 'head_hidden'):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if (self.input_factors, self.quant_window, self.d_model, self.head_hidden) != (147,12,128,64):
            raise ValueError('week-2 contract requires 12x147 input, 128-d encoding and 64-d head')
        if self.d_model % self.latent_groups or self.latent_groups in (1, self.d_model):
            raise ValueError('latent_groups must divide d_model, with both axes larger than one')
        if self.kernel_size % 2 == 0 or self.kernel_size > self.quant_window:
            raise ValueError('kernel must be odd and <= quant_window')
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError('dropout must be in [0,1)')
        if self.pooling not in ('last','mean'):
            raise ValueError('pooling must be last or mean')


class ModernTCNBlock(nn.Module):
    """[B,M,D,T] -> [B,M,D,T], total width M*D = 128 by default.

    Depthwise temporal convolution; LayerNorm over each group's D features;
    grouped pointwise FFN on D; permute and grouped pointwise FFN on M;
    then one residual addition. LayerNorm avoids cross-sample batch statistics.
    """
    def __init__(self, groups=16, features=8, kernel_size=7, ffn_ratio=2, dropout=0.1):
        super().__init__()
        if min(groups, features, ffn_ratio, kernel_size) < 1 or kernel_size % 2 == 0:
            raise ValueError('positive dimensions and odd kernel required')
        self.groups, self.features = groups, features
        width = groups * features
        self.temporal = nn.Conv1d(width, width, kernel_size, padding=kernel_size//2, groups=width)
        self.norm = nn.LayerNorm(features)
        self.feature_ffn = nn.Sequential(
            nn.Conv1d(width, width*ffn_ratio, 1, groups=groups),
            nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(width*ffn_ratio, width, 1, groups=groups), nn.Dropout(dropout))
        self.variable_ffn = nn.Sequential(
            nn.Conv1d(width, width*ffn_ratio, 1, groups=features),
            nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(width*ffn_ratio, width, 1, groups=features), nn.Dropout(dropout))

    def forward(self, value):
        if value.ndim != 4 or tuple(value.shape[1:3]) != (self.groups, self.features):
            raise ValueError('block expects [B,latent_groups,features,T]')
        batch, groups, features, time = value.shape
        mixed = self.temporal(value.reshape(batch, groups*features, time))
        mixed = mixed.reshape(batch, groups, features, time).permute(0,1,3,2)
        mixed = self.norm(mixed).permute(0,1,3,2)
        mixed = self.feature_ffn(mixed.reshape(batch, groups*features, time))
        mixed = mixed.reshape(batch, groups, features, time).permute(0,2,1,3)
        mixed = self.variable_ffn(mixed.reshape(batch, features*groups, time))
        mixed = mixed.reshape(batch, features, groups, time).permute(0,2,1,3)
        return value + mixed


class ModernTCNEncoder(nn.Module):
    """Shared numerical encoder: [B,12,147] -> [B,128]."""
    def __init__(self, config=None):
        super().__init__()
        self.config = config or QuantModelConfig()
        c = self.config
        self.projection = nn.Linear(c.input_factors, c.d_model)
        self.blocks = nn.Sequential(*[
            ModernTCNBlock(c.latent_groups, c.d_model//c.latent_groups,
                           c.kernel_size, c.ffn_ratio, c.dropout)
            for _ in range(c.num_blocks)])
        self.output_norm = nn.LayerNorm(c.d_model)

    def forward(self, quant):
        c = self.config
        if quant.ndim != 3 or tuple(quant.shape[1:]) != (c.quant_window,c.input_factors) or quant.shape[0] < 1:
            raise ValueError('encoder expects nonempty [B,12,147]')
        if not quant.is_floating_point() or not torch.isfinite(quant).all():
            raise ValueError('quant must be finite floating point')
        batch = quant.shape[0]
        value = self.projection(quant).transpose(1,2)
        value = value.reshape(batch, c.latent_groups, c.d_model//c.latent_groups, c.quant_window)
        value = self.blocks(value).reshape(batch, c.d_model, c.quant_window)
        pooled = value[:,:,-1] if c.pooling == 'last' else value.mean(dim=-1)
        return self.output_norm(pooled)
