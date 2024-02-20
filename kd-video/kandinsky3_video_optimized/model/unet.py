# This source code is licensed under the Apache License found in the
# LICENSE file in the current directory.
"""
The code has been adopted from KandinskyVideo
(https://github.com/ai-forever/KandinskyVideo/blob/main/video_kandinsky3/model/unet.py)
"""

import torch
from torch import nn, einsum
from einops import rearrange
from torch.nn.parameter import Parameter

from .nn import (
    Identity,
    Attention,
    SinusoidalPosEmb,
    ConditionalGroupNorm,
    TemporalAttention,
)
from .utils import exist, set_default_item, set_default_layer


class Block(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        time_embed_dim,
        kernel_size=3,
        norm_groups=32,
        up_resolution=None,
        temporal=False,
    ):
        super().__init__()
        self.group_norm = ConditionalGroupNorm(norm_groups, in_channels, time_embed_dim)
        self.activation = nn.SiLU()
        self.up_sample = set_default_layer(
            exist(up_resolution) and up_resolution,
            nn.ConvTranspose2d,
            (in_channels, in_channels),
            {"kernel_size": 2, "stride": 2},
        )
        padding = set_default_item(kernel_size == 1, 0, 1)
        self.projection = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=padding
        )
        self.down_sample = set_default_layer(
            exist(up_resolution) and not up_resolution,
            nn.Conv2d,
            (out_channels, out_channels),
            {"kernel_size": 2, "stride": 2},
        )

        if temporal:
            self.temporal_merge = torch.nn.Parameter(torch.tensor([0.0]))
            self.temporal_projection = nn.Conv3d(
                out_channels, out_channels, (3, 1, 1), padding=(1, 0, 0)
            )
            self.temporal_projection.weight.data.zero_()
            self.temporal_projection.bias.data.zero_()

    def forward(self, x, time_embed, num_temporal_groups=None):
        x = self.group_norm(x, time_embed)
        x = self.activation(x)
        x = self.up_sample(x)
        x = self.projection(x)
        if exist(num_temporal_groups):
            out = rearrange(x, "(b t) c h w -> b c t h w", t=num_temporal_groups)
            out = self.temporal_projection(out)
            out = rearrange(out, "b c t h w -> (b t) c h w")

            temporal_merge = self.temporal_merge.to(dtype=torch.float16)

            x = (1 - temporal_merge) * x + temporal_merge * out
        x = self.down_sample(x)
        return x


class ResNetBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        time_embed_dim,
        norm_groups=32,
        compression_ratio=2,
        up_resolutions=None,
        temporal=False,
    ):
        super().__init__()
        kernel_sizes = [1, 3, 3, 1]
        hidden_channel = max(in_channels, out_channels) // compression_ratio
        hidden_channels = (
            [(in_channels, hidden_channel)]
            + [(hidden_channel, hidden_channel)] * 2
            + [(hidden_channel, out_channels)]
        )
        if not exist(up_resolutions):
            up_resolutions = 4 * [None]
        self.resnet_blocks = nn.ModuleList(
            [
                Block(
                    in_channel,
                    out_channel,
                    time_embed_dim,
                    kernel_size,
                    norm_groups,
                    up_resolution,
                    temporal,
                )
                for (in_channel, out_channel), kernel_size, up_resolution in zip(
                    hidden_channels, kernel_sizes, up_resolutions
                )
            ]
        )

        self.shortcut_up_sample = set_default_layer(
            True in up_resolutions,
            nn.ConvTranspose2d,
            (in_channels, in_channels),
            {"kernel_size": 2, "stride": 2},
        )
        self.shortcut_projection = set_default_layer(
            in_channels != out_channels,
            nn.Conv2d,
            (in_channels, out_channels),
            {"kernel_size": 1},
        )
        self.shortcut_down_sample = set_default_layer(
            False in up_resolutions,
            nn.Conv2d,
            (out_channels, out_channels),
            {"kernel_size": 2, "stride": 2},
        )

    def forward(self, x, time_embed, num_temporal_groups=None):
        out = x
        for resnet_block in self.resnet_blocks:
            out = resnet_block(out, time_embed, num_temporal_groups)

        x = self.shortcut_up_sample(x)
        x = self.shortcut_projection(x)
        x = self.shortcut_down_sample(x)
        x = x + out
        return x


class TemporalResNetBlock(nn.Module):

    def __init__(
        self,
        num_channels,
        time_embed_dim,
        num_frames,
        norm_groups=32,
        compression_ratio=2,
    ):
        super().__init__()
        self.num_frames = num_frames
        kernel_sizes = [1, 3, 3, 1]
        paddings = [0, 1, 1, 0]
        hidden_channel = num_channels // compression_ratio
        hidden_channels = (
            [(num_channels, hidden_channel)]
            + [(hidden_channel, hidden_channel)] * 2
            + [(hidden_channel, num_channels)]
        )
        self.temporal_resnet_blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        ConditionalGroupNorm(norm_groups, in_channel, time_embed_dim),
                        nn.SiLU(),
                        nn.Conv3d(
                            in_channel,
                            out_channel,
                            (kernel_size, 1, 1),
                            padding=(padding, 0, 0),
                        ),
                    ]
                )
                for (in_channel, out_channel), kernel_size, padding in zip(
                    hidden_channels, kernel_sizes, paddings
                )
            ]
        )
        nn.init.zeros_(self.temporal_resnet_blocks[-1][-1].weight)
        nn.init.zeros_(self.temporal_resnet_blocks[-1][-1].bias)

    def forward(self, x, time_embed):
        out = x
        for group_norm, activation, projection in self.temporal_resnet_blocks:
            out = group_norm(out, time_embed)
            out = activation(out)
            out = rearrange(out, "(b t) c h w -> b c t h w", t=self.num_frames)
            out = projection(out)
            out = rearrange(out, "b c t h w -> (b t) c h w", t=self.num_frames)
        x = x + out
        return x


class AttentionPolling(nn.Module):

    def __init__(self, num_channels, context_dim, head_dim=64):
        super().__init__()
        self.attention = Attention(context_dim, num_channels, context_dim, head_dim)

    def forward(self, x, context, context_mask=None):
        context = self.attention(
            context.mean(dim=1, keepdim=True), context, context_mask
        )
        return x + context.squeeze(1)


class AttentionBlock(nn.Module):

    def __init__(
        self,
        num_channels,
        time_embed_dim,
        context_dim=None,
        norm_groups=32,
        head_dim=64,
        expansion_ratio=4,
    ):
        super().__init__()
        self.in_norm = ConditionalGroupNorm(norm_groups, num_channels, time_embed_dim)
        self.attention = Attention(
            num_channels, num_channels, context_dim or num_channels, head_dim
        )

        hidden_channels = expansion_ratio * num_channels
        self.out_norm = ConditionalGroupNorm(norm_groups, num_channels, time_embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Conv2d(num_channels, hidden_channels, kernel_size=1, bias=False),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, num_channels, kernel_size=1, bias=False),
        )

    def forward(self, x, time_embed, context=None, context_mask=None):
        height, width = x.shape[-2:]
        out = self.in_norm(x, time_embed)
        out = rearrange(out, "b c h w -> b (h w) c", h=height, w=width)
        context = set_default_item(exist(context), context, out)
        out = self.attention(out, context, context_mask)
        out = rearrange(out, "b (h w) c -> b c h w", h=height, w=width)
        x = x + out

        out = self.out_norm(x, time_embed)
        out = self.feed_forward(out)
        x = x + out
        return x


class TemporalAttentionBlock(nn.Module):

    def __init__(
        self,
        num_channels,
        time_embed_dim,
        num_frames,
        norm_groups=32,
        head_dim=64,
        expansion_ratio=4,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.in_norm = ConditionalGroupNorm(norm_groups, num_channels, time_embed_dim)
        self.temporal_attention = TemporalAttention(
            num_channels, num_channels, head_dim
        )

        hidden_channels = expansion_ratio * num_channels
        self.out_norm = ConditionalGroupNorm(norm_groups, num_channels, time_embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Conv2d(num_channels, hidden_channels, kernel_size=1, bias=False),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, num_channels, kernel_size=1, bias=False),
        )

    def forward(self, x, time_embed, temporal_embed):
        height, width = x.shape[-2:]
        out = self.in_norm(x, temporal_embed)
        out = rearrange(
            out, "(b t) c h w -> b t (h w) c", h=height, w=width, t=self.num_frames
        )
        out = self.temporal_attention(out)
        out = rearrange(
            out, "b t (h w) c -> (b t) c h w", h=height, w=width, t=self.num_frames
        )
        x = x + out

        out = self.out_norm(x, time_embed)
        out = self.feed_forward(out)
        x = x + out
        return x


class DownSampleBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        time_embed_dim,
        context_dim=None,
        num_blocks=3,
        groups=32,
        head_dim=64,
        expansion_ratio=4,
        compression_ratio=2,
        down_sample=True,
        self_attention=True,
        num_frames=None,
        interpolation=False,
    ):
        super().__init__()
        self.self_attention_block = set_default_layer(
            self_attention,
            AttentionBlock,
            (in_channels, time_embed_dim, None, groups, head_dim, expansion_ratio),
            layer_2=Identity,
        )
        self.temporal_attention_block = set_default_layer(
            exist(num_frames) and self_attention,
            TemporalAttentionBlock,
            (
                in_channels,
                time_embed_dim,
                num_frames,
                groups,
                head_dim,
                expansion_ratio,
            ),
            layer_2=Identity,
        )

        up_resolutions = [[None] * 4] * (num_blocks - 1) + [
            [None, None, set_default_item(down_sample, False), None]
        ]
        hidden_channels = [(in_channels, out_channels)] + [
            (out_channels, out_channels)
        ] * (num_blocks - 1)
        self.resnet_attn_blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        ResNetBlock(
                            in_channel,
                            out_channel,
                            time_embed_dim,
                            groups,
                            compression_ratio,
                            temporal=interpolation,
                        ),
                        set_default_layer(
                            exist(context_dim),
                            AttentionBlock,
                            (
                                out_channel,
                                time_embed_dim,
                                context_dim,
                                groups,
                                head_dim,
                                expansion_ratio,
                            ),
                            layer_2=Identity,
                        ),
                        ResNetBlock(
                            out_channel,
                            out_channel,
                            time_embed_dim,
                            groups,
                            compression_ratio,
                            up_resolution,
                            temporal=interpolation,
                        ),
                        set_default_layer(
                            exist(num_frames),
                            TemporalResNetBlock,
                            (
                                out_channel,
                                time_embed_dim,
                                num_frames,
                                groups,
                                compression_ratio,
                            ),
                            layer_2=Identity,
                        ),
                        set_default_layer(
                            exist(num_frames),
                            TemporalResNetBlock,
                            (
                                out_channel,
                                time_embed_dim,
                                num_frames,
                                groups,
                                compression_ratio,
                            ),
                            layer_2=Identity,
                        ),
                    ]
                )
                for (in_channel, out_channel), up_resolution in zip(
                    hidden_channels, up_resolutions
                )
            ]
        )

    def forward(
        self,
        x,
        time_embed,
        context=None,
        context_mask=None,
        temporal_embed=None,
        num_temporal_groups=None,
    ):
        x = self.self_attention_block(x, time_embed)
        x = self.temporal_attention_block(x, time_embed, temporal_embed)
        for (
            in_resnet_block,
            attention,
            out_resnet_block,
            in_temporal_resnet_block,
            out_temporal_resnet_block,
        ) in self.resnet_attn_blocks:
            x = in_resnet_block(x, time_embed, num_temporal_groups)
            x = in_temporal_resnet_block(x, time_embed)
            x = attention(x, time_embed, context, context_mask)
            x = out_resnet_block(x, time_embed, num_temporal_groups)
            x = out_temporal_resnet_block(x, time_embed)
        return x


class UpSampleBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        cat_dim,
        out_channels,
        time_embed_dim,
        context_dim=None,
        num_blocks=3,
        groups=32,
        head_dim=64,
        expansion_ratio=4,
        compression_ratio=2,
        up_sample=True,
        self_attention=True,
        num_frames=None,
        interpolation=False,
    ):
        super().__init__()
        up_resolutions = [[None, set_default_item(up_sample, True), None, None]] + [
            [None] * 4
        ] * (num_blocks - 1)
        hidden_channels = (
            [(in_channels + cat_dim, in_channels)]
            + [(in_channels, in_channels)] * (num_blocks - 2)
            + [(in_channels, out_channels)]
        )
        self.resnet_attn_blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        ResNetBlock(
                            in_channel,
                            in_channel,
                            time_embed_dim,
                            groups,
                            compression_ratio,
                            up_resolution,
                            temporal=interpolation,
                        ),
                        set_default_layer(
                            exist(context_dim),
                            AttentionBlock,
                            (
                                in_channel,
                                time_embed_dim,
                                context_dim,
                                groups,
                                head_dim,
                                expansion_ratio,
                            ),
                            layer_2=Identity,
                        ),
                        ResNetBlock(
                            in_channel,
                            out_channel,
                            time_embed_dim,
                            groups,
                            compression_ratio,
                            temporal=interpolation,
                        ),
                        set_default_layer(
                            exist(num_frames),
                            TemporalResNetBlock,
                            (
                                in_channel,
                                time_embed_dim,
                                num_frames,
                                groups,
                                compression_ratio,
                            ),
                            layer_2=Identity,
                        ),
                        set_default_layer(
                            exist(num_frames),
                            TemporalResNetBlock,
                            (
                                in_channel,
                                time_embed_dim,
                                num_frames,
                                groups,
                                compression_ratio,
                            ),
                            layer_2=Identity,
                        ),
                    ]
                )
                for (in_channel, out_channel), up_resolution in zip(
                    hidden_channels, up_resolutions
                )
            ]
        )

        self.temporal_attention_block = set_default_layer(
            exist(num_frames) and self_attention,
            TemporalAttentionBlock,
            (
                out_channels,
                time_embed_dim,
                num_frames,
                groups,
                head_dim,
                expansion_ratio,
            ),
            layer_2=Identity,
        )
        self.self_attention_block = set_default_layer(
            self_attention,
            AttentionBlock,
            (out_channels, time_embed_dim, None, groups, head_dim, expansion_ratio),
            layer_2=Identity,
        )

    def forward(
        self,
        x,
        time_embed,
        context=None,
        context_mask=None,
        temporal_embed=None,
        num_temporal_groups=None,
    ):
        for (
            in_resnet_block,
            attention,
            out_resnet_block,
            in_temporal_resnet_block,
            out_temporal_resnet_block,
        ) in self.resnet_attn_blocks:
            x = in_temporal_resnet_block(x, time_embed)
            x = in_resnet_block(x, time_embed, num_temporal_groups)
            x = attention(x, time_embed, context, context_mask)
            x = out_temporal_resnet_block(x, time_embed)
            x = out_resnet_block(x, time_embed, num_temporal_groups)
        x = self.temporal_attention_block(x, time_embed, temporal_embed)
        x = self.self_attention_block(x, time_embed)
        return x


class UNet(nn.Module):

    def __init__(
        self,
        model_channels,
        init_channels=None,
        num_channels=3,
        out_channels=3,
        time_embed_dim=None,
        context_dim=None,
        groups=32,
        head_dim=64,
        expansion_ratio=4,
        compression_ratio=2,
        dim_mult=(1, 2, 4, 8),
        num_blocks=(3, 3, 3, 3),
        add_cross_attention=(False, True, True, True),
        add_self_attention=(False, True, True, True),
        num_frames=None,
        interpolation=False,
    ):
        super().__init__()
        num_frames = set_default_item(interpolation, None, num_frames)
        self.num_frames = num_frames
        init_channels = init_channels or model_channels
        self.to_time_embed = nn.Sequential(
            SinusoidalPosEmb(init_channels),
            nn.Linear(init_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        if exist(context_dim):
            self.feature_pooling = AttentionPolling(
                time_embed_dim, context_dim, head_dim
            )
        if exist(num_frames):
            self.to_temporal_embed = nn.Sequential(
                SinusoidalPosEmb(init_channels),
                nn.Linear(init_channels, time_embed_dim),
                nn.SiLU(),
                nn.Linear(time_embed_dim, time_embed_dim),
            )
        if interpolation:
            self.perturbation_to_time_embed = nn.Sequential(
                SinusoidalPosEmb(init_channels),
                nn.Linear(init_channels, time_embed_dim),
                nn.SiLU(),
                nn.Linear(time_embed_dim, time_embed_dim),
            )
            self.time_embed_merge = torch.nn.Parameter(torch.tensor([0.0]))

            self.skip_merge = torch.nn.Parameter(torch.tensor([0.0]))

            self.skip_embeddings = torch.nn.Embedding(20, time_embed_dim)

        self.in_layer = nn.Conv2d(num_channels, init_channels, kernel_size=3, padding=1)
        hidden_dims = [
            init_channels,
            *map(lambda mult: model_channels * mult, dim_mult),
        ]
        in_out_dims = list(zip(hidden_dims[:-1], hidden_dims[1:]))
        text_dims = [
            set_default_item(is_exist, context_dim) for is_exist in add_cross_attention
        ]
        layer_params = [num_blocks, text_dims, add_self_attention]
        rev_layer_params = map(reversed, layer_params)

        cat_dims = []
        self.num_levels = len(in_out_dims)
        self.down_samples = nn.ModuleList([])
        for level, (
            (in_dim, out_dim),
            res_block_num,
            text_dim,
            self_attention,
        ) in enumerate(zip(in_out_dims, *layer_params)):
            down_sample = level != (self.num_levels - 1)
            cat_dims.append(
                set_default_item(level != (self.num_levels - 1), out_dim, 0)
            )
            self.down_samples.append(
                DownSampleBlock(
                    in_dim,
                    out_dim,
                    time_embed_dim,
                    text_dim,
                    res_block_num,
                    groups,
                    head_dim,
                    expansion_ratio,
                    compression_ratio,
                    down_sample,
                    self_attention,
                    num_frames,
                    interpolation,
                )
            )

        self.up_samples = nn.ModuleList([])
        for level, (
            (out_dim, in_dim),
            res_block_num,
            text_dim,
            self_attention,
        ) in enumerate(zip(reversed(in_out_dims), *rev_layer_params)):
            up_sample = level != 0
            self.up_samples.append(
                UpSampleBlock(
                    in_dim,
                    cat_dims.pop(),
                    out_dim,
                    time_embed_dim,
                    text_dim,
                    res_block_num,
                    groups,
                    head_dim,
                    expansion_ratio,
                    compression_ratio,
                    up_sample,
                    self_attention,
                    num_frames,
                    interpolation,
                )
            )

        self.out_layer = nn.Sequential(
            nn.GroupNorm(groups, init_channels),
            nn.SiLU(),
            nn.Conv2d(init_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(
        self,
        x,
        time,
        context=None,
        context_mask=None,
        temporal_embed=None,
        skip_frames=None,
        num_temporal_groups=None,
    ):
        time_embed = self.to_time_embed(time)
        if exist(context):
            time_embed = self.feature_pooling(time_embed, context, context_mask)

        hidden_states = []
        x = self.in_layer(x)
        if exist(temporal_embed):
            temporal_embed = self.to_temporal_embed(temporal_embed)
        if exist(num_temporal_groups):
            perturbation_time = torch.zeros(
                size=(context.shape[0],), device=context.device
            )
            perturbation_time_embed = self.perturbation_to_time_embed(perturbation_time)
            time_embed_merge = self.time_embed_merge.to(dtype=torch.float16)

            time_embed = (
                time_embed_merge * perturbation_time_embed
                + (1 - time_embed_merge) * time_embed
            )

            self.skip_embeddings.weight = Parameter(
                self.skip_embeddings.weight.to(dtype=torch.float16), requires_grad=False
            )

            skip_emb = self.skip_embeddings(skip_frames.to(device="cuda"))

            skip_merge = self.skip_merge.to(dtype=torch.float16)
            time_embed = skip_merge * skip_emb + time_embed

        for level, down_sample in enumerate(self.down_samples):
            x = down_sample(
                x,
                time_embed,
                context,
                context_mask,
                temporal_embed,
                num_temporal_groups,
            )
            if level != self.num_levels - 1:
                hidden_states.append(x)
        for level, up_sample in enumerate(self.up_samples):
            if level != 0:
                x = torch.cat([x, hidden_states.pop()], dim=1)
            x = up_sample(
                x,
                time_embed,
                context,
                context_mask,
                temporal_embed,
                num_temporal_groups,
            )
        x = self.out_layer(x)
        return x


from torch.nn import functional
from typing import Optional

Tensor = torch.Tensor

_old_func = functional.embedding


def patch_functional():
    functional.embedding = patched_embedding


def unpatch_functional():
    functional.embedding = _old_func


def patched_embedding(
    input: Tensor,
    weight: Tensor,
    padding_idx: Optional[int] = None,
    max_norm: Optional[float] = None,
    norm_type: float = 2.0,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
) -> Tensor:
    input = input.type(torch.LongTensor).to(device="cuda")
    weight = weight.type(torch.LongTensor).to(device="cuda")

    _old_func(
        input, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse
    )


def get_unet(conf):
    unet = UNet(**conf)
    return unet
