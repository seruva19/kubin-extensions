# This source code is licensed under the Apache License found in the
# LICENSE file in the current directory.
"""
The code has been adopted from KandinskyVideo
(https://github.com/ai-forever/KandinskyVideo/blob/main/video_kandinsky3/__init__.py)
"""

from typing import Optional, Union, List, no_type_check, cast

import PIL
import os

import numpy as np
from tqdm import tqdm
from huggingface_hub import hf_hub_download

import torch
import omegaconf
from omegaconf import OmegaConf
from .model.unet import UNet
from .movq import MoVQ
from .condition_encoders import T5TextConditionEncoder
from .condition_processors import T5TextConditionProcessor
from .model.diffusion import BaseDiffusion, get_named_beta_schedule
from .utils import flush
from .t2v_pipeline import VideoKandinsky3T2VPipeline


@no_type_check
def get_T2V_unet(
    device: Union[str, torch.device],
    weights_path: Optional[str] = None,
    fp16: bool = False,
    fp8: bool = False,
) -> UNet:
    unet = UNet(
        model_channels=384,
        num_channels=4,
        out_channels=4,
        init_channels=192,
        time_embed_dim=1536,
        context_dim=4096,
        groups=32,
        head_dim=64,
        expansion_ratio=4,
        compression_ratio=2,
        dim_mult=(1, 2, 4, 8),
        num_blocks=(3, 3, 3, 3),
        add_cross_attention=(False, True, True, True),
        add_self_attention=(False, True, True, True),
        num_frames=15,
    )

    state_dict = torch.load(weights_path, map_location=torch.device("cpu"))
    unet.load_state_dict(state_dict["unet"])

    if fp16 and not fp8:
        unet.eval().to(device)
        unet = unet.half()

    else:
        if fp8:
            unet.eval().to(cast(torch.device, device), torch.float8_e4m3fn)
        elif fp16:
            unet.eval().to(cast(torch.device, device), torch.bfloat16)
        else:
            unet.eval().to(cast(torch.device, device))

    return unet


@no_type_check
def get_interpolation_unet(
    device: Union[str, torch.device],
    weights_path: Optional[str] = None,
    fp16: bool = False,
    fp8: bool = False,
) -> UNet:
    interpolation_unet = UNet(
        model_channels=384,
        num_channels=20,
        out_channels=12,
        init_channels=192,
        time_embed_dim=1536,
        context_dim=4096,
        groups=32,
        head_dim=64,
        expansion_ratio=4,
        compression_ratio=2,
        dim_mult=(1, 2, 4, 8),
        num_blocks=(3, 3, 3, 3),
        add_cross_attention=(False, True, True, True),
        add_self_attention=(False, True, True, True),
        interpolation=True,
    )

    state_dict = torch.load(weights_path, map_location=torch.device("cpu"))
    interpolation_unet.load_state_dict(state_dict["unet"])

    if fp16 and not fp8:
        interpolation_unet.eval().to(device)
        interpolation_unet = interpolation_unet.half()
    else:
        if fp8:
            interpolation_unet.eval().to(
                cast(torch.device, device), torch.float8_e4m3fn
            )
        elif fp16:
            interpolation_unet.eval().to(cast(torch.device, device), torch.bfloat16)
        else:
            interpolation_unet.eval().to(cast(torch.device, device))

    return interpolation_unet


@no_type_check
def get_unet_nullemb_projections(
    weights_path,
) -> (torch.Tensor, dict):
    state_dict = torch.load(weights_path, map_location=torch.device("cpu"))

    projections_state_dict = state_dict["projections"]
    null_embedding = state_dict["null_embedding"]

    return null_embedding, projections_state_dict


@no_type_check
def get_interpolation_unet_nullemb(
    weights_path,
) -> torch.Tensor:
    state_dict = torch.load(weights_path, map_location=torch.device("cpu"))

    null_embedding = state_dict["null_embedding"]

    return null_embedding


@no_type_check
def get_T5encoder(
    device: Union[str, torch.device],
    weights_path: str,
    projections_state_dict: Optional[dict] = None,
    fp16: bool = True,
    low_cpu_mem_usage: bool = True,
    device_map: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> (UNet, Optional[dict], Optional[torch.Tensor]):
    model_names = {"t5": weights_path}
    tokens_length = {"t5": 128}
    context_dim = 4096
    model_dims = {"t5": 4096}
    processor = T5TextConditionProcessor(
        tokens_length=tokens_length, processor_names=model_names, cache_dir=cache_dir
    )
    condition_encoders = T5TextConditionEncoder(
        model_names,
        context_dim,
        model_dims,
        low_cpu_mem_usage=low_cpu_mem_usage,
        device_map=device_map,
        cache_dir=cache_dir,
    )

    if projections_state_dict:
        condition_encoders.projections.load_state_dict(projections_state_dict)

    condition_encoders = condition_encoders.eval().to(device)
    return processor, condition_encoders


def get_T5processor(weights_path: str, cache_dir: str) -> T5TextConditionProcessor:
    model_names = {"t5": weights_path}
    tokens_length = {"t5": 128}

    processor = T5TextConditionProcessor(
        tokens_length=tokens_length, processor_names=model_names, cache_dir=cache_dir
    )

    return processor


def get_movq(
    device: Union[str, torch.device],
    weights_path: str,
    fp16: bool = False,
    fp8: bool = False,
) -> MoVQ:
    generator_config = {
        "double_z": False,
        "z_channels": 4,
        "resolution": 256,
        "in_channels": 3,
        "out_ch": 3,
        "ch": 256,
        "ch_mult": [1, 2, 2, 4],
        "num_res_blocks": 2,
        "attn_resolutions": [32],
        "dropout": 0.0,
    }
    movq = MoVQ(generator_config)
    movq.load_state_dict(torch.load(weights_path))

    if fp8:
        movq = movq.eval().to(cast(torch.device, device), torch.float8_e4m3fn)
    elif fp16:
        movq = movq.eval().to(cast(torch.device, device), torch.bfloat16)
    else:
        movq = movq.eval().to(cast(torch.device, device))

    return movq


def get_T2V_pipeline(
    device: Union[str, torch.device],
    fp16: bool = False,
    fp8: bool = False,
    cache_dir: str = "/tmp/kandinsky_video/",
    unet_path: str | None = None,
    interpolation_unet_path: str | None = None,
    text_encode_path: str | None = None,
    movq_path: str | None = None,
) -> VideoKandinsky3T2VPipeline:
    if unet_path is None:
        unet_path = hf_hub_download(
            repo_id="ai-forever/KandinskyVideo",
            filename="weights/kandinsky_video.pt",
            cache_dir=cache_dir,
        )
    if interpolation_unet_path is None:
        interpolation_unet_path = hf_hub_download(
            repo_id="ai-forever/KandinskyVideo",
            filename="weights/kandinsky_video_interpolation.pt",
            cache_dir=cache_dir,
        )
    if text_encode_path is None:
        text_encode_path = "google/flan-ul2"
    if movq_path is None:
        movq_path = hf_hub_download(
            repo_id="ai-forever/KandinskyVideo",
            filename="weights/movq.pt",
            cache_dir=cache_dir,
        )

    null_embedding, projections_state_dict = get_unet_nullemb_projections(unet_path)
    interpolation_null_embedding = get_interpolation_unet_nullemb(
        interpolation_unet_path
    )
    processor = get_T5processor(text_encode_path, cache_dir)

    encoder_loader = lambda: get_T5encoder(
        device, text_encode_path, projections_state_dict, fp16=fp16, cache_dir=cache_dir
    )[1]
    unet_loader = lambda: get_T2V_unet(device, unet_path, fp16=fp16, fp8=fp8)
    interpolation_unet_loader = lambda: get_interpolation_unet(
        device, interpolation_unet_path, fp16=fp16, fp8=fp8
    )
    movq_loader = lambda: get_movq(device, movq_path, fp16=fp16, fp8=fp8)

    flush()

    return VideoKandinsky3T2VPipeline(
        device,
        unet_loader,
        null_embedding,
        interpolation_unet_loader,
        interpolation_null_embedding,
        processor,
        encoder_loader,
        movq_loader,
        fp16=fp16,
    )
