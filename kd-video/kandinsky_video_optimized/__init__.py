# This source code is licensed under the Apache License found in the
# LICENSE file in the current directory.
"""
The code has been adopted from KandinskyVideo
(https://github.com/ai-forever/KandinskyVideo/blob/main/kandinsky_video/__init__.py)
"""

import os
import json
from tqdm import tqdm
from typing import Optional, Union, cast

import torch
from huggingface_hub import hf_hub_download

from .model.unet import UNet
from .model.unet_interpolation import UNet as UNetInterpolation
from .movq import MoVQ
from .condition_encoders import T5TextConditionEncoder
from .condition_processors import T5TextConditionProcessor

from .t2v_pipeline import KandinskyVideoT2VPipeline

REPO_ID = "ai-forever/KandinskyVideo_1_1"


def get_T2V_unet(
    weights_path: str,
    device: Union[str, torch.device],
    configs: Optional[dict] = None,
) -> UNet:
    unet = UNet(**configs)

    state_dict = torch.load(weights_path, map_location=torch.device("cpu"))
    unet.load_state_dict(state_dict["unet"])

    unet.eval().to(cast(torch.device, device), torch.float8_e4m3fn)

    return unet


def get_interpolation_unet(
    weights_path: str,
    device: Union[str, torch.device],
    configs: Optional[dict] = None,
) -> UNet:
    interpolation_unet = UNetInterpolation(**configs)

    state_dict = torch.load(weights_path, map_location=torch.device("cpu"))
    interpolation_unet.load_state_dict(state_dict["unet"])

    interpolation_unet.eval().to(cast(torch.device, device), torch.float8_e4m3fn)

    return interpolation_unet


def get_unet_nullemb(
    weights_path,
) -> torch.Tensor:
    state_dict = torch.load(weights_path, map_location=torch.device("cpu"))
    null_embedding = state_dict["null_embedding"]

    return null_embedding


def get_T5encoder(
    cache_dir: str,
    device: Union[str, torch.device],
    weights_path: str,
    tokens_length: int = 128,
    context_dim: int = 4096,
    dtype: Union[str, torch.dtype] = torch.float32,
    low_cpu_mem_usage: bool = True,
) -> (T5TextConditionProcessor, T5TextConditionEncoder):  # type: ignore
    t5_projections_path = hf_hub_download(
        repo_id=REPO_ID, filename="t5_projections.pt", cache_dir=cache_dir
    )

    condition_encoder = T5TextConditionEncoder(
        weights_path,
        t5_projections_path,
        context_dim,
        cache_dir,
        low_cpu_mem_usage=low_cpu_mem_usage,
        dtype=dtype,
    )

    return condition_encoder


def get_T5processor(
    weights_path: str, cache_dir: str, tokens_length: int = 128
) -> T5TextConditionProcessor:
    processor = T5TextConditionProcessor(tokens_length, weights_path, cache_dir)

    return processor


def get_movq(
    cache_dir: str, device: Union[str, torch.device], configs: dict = None
) -> MoVQ:
    weights_path = hf_hub_download(
        repo_id=REPO_ID, filename="movq.pt", cache_dir=cache_dir
    )

    movq = MoVQ(configs)

    state_dict = torch.load(weights_path, map_location=torch.device("cpu"))
    movq.load_state_dict(state_dict)

    movq = movq.eval().to(cast(torch.device, device), torch.float8_e4m3fn)
    return movq


def get_video_movq(
    cache_dir: str,
    device: Union[str, torch.device],
    configs: dict = None,
    dtype: Union[str, torch.dtype] = torch.float32,
) -> MoVQ:
    weights_path = hf_hub_download(
        repo_id=REPO_ID, filename="video_movq.pt", cache_dir=cache_dir
    )

    video_movq = MoVQ(configs)

    state_dict = torch.load(weights_path, map_location=torch.device("cpu"))
    video_movq.load_state_dict(state_dict)

    video_movq = video_movq.eval().to(cast(torch.device, device), torch.float8_e4m3fn)
    return video_movq


def get_T2V_pipeline(
    device_map: Union[str, torch.device, dict],
    cache_dir: str = "/tmp/kandinsky_video/",
    text_encoder_path: str | None = None,
) -> KandinskyVideoT2VPipeline:

    configs_path = hf_hub_download(
        repo_id=REPO_ID, filename="configs.json", cache_dir=cache_dir
    )
    configs = json.load(open(configs_path))

    if not isinstance(device_map, dict):
        device_map = {
            "unet": device_map,
            "interpolation_unet": device_map,
            "text_encoder": device_map,
            "movq": device_map,
        }

    dtype_map = {
        "unet": torch.float16,
        "interpolation_unet": torch.float16,
        "text_encoder": torch.float32,
        "movq": torch.float32,
    }

    unet_weights_path = hf_hub_download(
        repo_id=REPO_ID, filename="t2v.pt", cache_dir=cache_dir
    )

    unet_null_embedding = get_unet_nullemb(unet_weights_path)

    unet_loader = lambda: get_T2V_unet(
        unet_weights_path, device_map["unet"], **configs["t2v"]
    )

    interpolation_unet_weights_path = hf_hub_download(
        repo_id=REPO_ID, filename="interpolation.pt", cache_dir=cache_dir
    )

    interpolation_null_embedding = get_unet_nullemb(interpolation_unet_weights_path)

    interpolation_unet_loader = lambda: get_interpolation_unet(
        interpolation_unet_weights_path,
        device_map["interpolation_unet"],
        **configs["interpolation"]
    )

    processor = get_T5processor(text_encoder_path, cache_dir)

    condition_encoder_loader = lambda: get_T5encoder(
        cache_dir=cache_dir,
        device=device_map["text_encoder"],
        weights_path=text_encoder_path,
        # **configs["text_encoder"]
    )

    movq_loader = lambda: get_movq(
        cache_dir, device_map["movq"], **configs["image_movq"]
    )
    video_movq_loader = lambda: get_video_movq(
        cache_dir, device_map["movq"], dtype=dtype_map["movq"], **configs["video_movq"]
    )
    return KandinskyVideoT2VPipeline(
        "cuda",
        dtype_map,
        unet_loader,
        unet_null_embedding,
        interpolation_unet_loader,
        interpolation_null_embedding,
        processor,
        condition_encoder_loader,
        movq_loader,
        video_movq_loader,
    )
