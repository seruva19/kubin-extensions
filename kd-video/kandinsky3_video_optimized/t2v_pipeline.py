# This source code is licensed under the Apache License found in the
# LICENSE file in the current directory.
"""
The code has been adopted from KandinskyVideo
(https://github.com/ai-forever/KandinskyVideo/blob/main/video_kandinsky3/t2v_pipeline.py)
"""

from typing import Callable, List, Union
import PIL
import random

import numpy as np

import torch
import torchvision.transforms as T
from einops import repeat, rearrange

from .model.unet import UNet
from .movq import MoVQ
from .condition_encoders import T5TextConditionEncoder
from .condition_processors import T5TextConditionProcessor
from .model.diffusion import BaseDiffusion, get_named_beta_schedule
from .utils import flush, vram_info


class VideoKandinsky3T2VPipeline:

    def __init__(
        self,
        device: Union[str, torch.device],
        unet_loader: Callable[[], UNet],
        null_embedding: torch.Tensor,
        interpolation_unet_loader: Callable[[], UNet],
        interpolation_null_embedding: torch.Tensor,
        t5_processor: T5TextConditionProcessor,
        t5_encoder_loader: Callable[[], T5TextConditionEncoder],
        movq_loader: Callable[[], MoVQ],
        fp16: bool = True,
    ):
        self.device = device
        self.fp16 = fp16
        self.to_pil = T.ToPILImage()

        self.unet_loader = unet_loader
        self.null_embedding = null_embedding

        self.interpolation_unet_loader = interpolation_unet_loader
        self.interpolation_null_embedding = interpolation_null_embedding

        self.t5_processor = t5_processor
        self.t5_encoder_loader = t5_encoder_loader
        self.movq_loader = movq_loader

    def _reshape_temporal_groups(self, temporal_groups, video):
        temporal_groups = rearrange(temporal_groups, "b (t c) h w -> b t c h w", t=3)
        temporal_groups = torch.cat(
            [temporal_groups[i] for i in range(temporal_groups.shape[0])], axis=0
        )

        b, c, h, w = video.shape
        video_upsampled = torch.zeros((4 * b - 3, c, h, w), device=self.device)

        interpolation_indices = [
            i for i in range(video_upsampled.shape[0]) if i % 4 != 0
        ]
        keyframes_indices = [i for i in range(video_upsampled.shape[0]) if i % 4 == 0]

        video_upsampled[interpolation_indices] = temporal_groups
        video_upsampled[keyframes_indices] = video
        return video_upsampled

    def generate_base_frames(
        self,
        base_diffusion,
        height,
        width,
        guidance_scale,
        condition_model_input,
        negative_condition_model_input=None,
    ):
        self.t5_encoder = self.t5_encoder_loader()
        vram_info("t5_encoder loaded")

        context, context_mask = self.t5_encoder(condition_model_input)
        if negative_condition_model_input is not None:
            negative_context, negative_context_mask = self.t5_encoder(
                negative_condition_model_input
            )
        else:
            negative_context, negative_context_mask = None, None

        self.t5_encoder.to("cpu")
        self.t5_encoder = None
        flush()
        vram_info("t5_encoder flushed")

        self.unet = self.unet_loader()
        vram_info("unet loaded")

        bs_context = repeat(context, "1 n d -> b n d", b=self.unet.num_frames)
        bs_context_mask = repeat(context_mask, "1 n -> b n", b=self.unet.num_frames)
        if negative_context is not None:
            bs_negative_context = repeat(
                negative_context, "1 n d -> b n d", b=self.unet.num_frames
            )
            bs_negative_context_mask = repeat(
                negative_context_mask, "1 n -> b n", b=self.unet.num_frames
            )
        else:
            bs_negative_context, bs_negative_context_mask = None, None

        video_len = 180
        temporal_positions = torch.arange(
            0, video_len, video_len // self.unet.num_frames, device=self.device
        )
        base_frames = base_diffusion.p_sample_loop(
            self.unet,
            (self.unet.num_frames, 4, height // 8, width // 8),
            self.device,
            bs_context,
            bs_context_mask,
            self.null_embedding,
            guidance_scale,
            temporal_positions=temporal_positions,
            negative_context=bs_negative_context,
            negative_context_mask=bs_negative_context_mask,
        )

        self.unet.to("cpu")
        self.unet = None
        flush()
        vram_info("unet flushed")

        return base_frames

    def interpolate_base_frames(
        self, base_diffusion, base_frames, height, width, guidance_scale, skip_frames
    ):
        num_temporal_groups = base_frames.shape[0] - 1
        left_base_frames, right_base_frames = base_frames[:-1], base_frames[1:]

        bs_context = torch.zeros([num_temporal_groups, 2, 4096], device=self.device)
        bs_context_mask = torch.zeros([num_temporal_groups, 2], device=self.device)
        bs_context[:, 0] = self.interpolation_null_embedding
        bs_context_mask[:, 0] = 1

        skip_frames = skip_frames * torch.ones(
            size=(num_temporal_groups,), dtype=torch.int32, device=self.device
        )

        self.interpolation_unet = self.interpolation_unet_loader()
        vram_info("interpolation_unet loaded")

        interpolated_base_frames = base_diffusion.p_sample_loop(
            self.interpolation_unet,
            (num_temporal_groups, 12, height // 8, width // 8),
            self.device,
            bs_context,
            bs_context_mask,
            self.interpolation_null_embedding,
            guidance_scale,
            base_frames=(left_base_frames, right_base_frames),
            num_temporal_groups=num_temporal_groups,
            skip_frames=skip_frames,
            v_predication=True,
        )

        self.interpolation_unet.to("cpu")
        self.interpolation_unet = None
        flush()
        vram_info("interpolation_unet flushed")

        return interpolated_base_frames

    def __call__(
        self,
        text: str,
        negative_text: str = None,
        width: int = 512,
        height: int = 512,
        fps: str = "low",
        guidance_scale: float = 5.0,
        interpolation_guidance_scale: float = 0.25,
        steps: int = 50,
        seed: int | None = None,
    ) -> List[PIL.Image.Image]:
        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)

        betas = get_named_beta_schedule("cosine", steps)
        base_diffusion = BaseDiffusion(betas, 0.98)

        condition_model_input, negative_condition_model_input = (
            self.t5_processor.encode(text, negative_text)
        )
        for key in condition_model_input:
            for input_type in condition_model_input[key]:
                condition_model_input[key][input_type] = (
                    condition_model_input[key][input_type].unsqueeze(0).to(self.device)
                )

        if negative_condition_model_input is not None:
            for key in negative_condition_model_input:
                for input_type in negative_condition_model_input[key]:
                    negative_condition_model_input[key][input_type] = (
                        negative_condition_model_input[key][input_type]
                        .unsqueeze(0)
                        .to(self.device)
                    )

        vram_info("init")

        pil_video = []
        with torch.autocast("cuda"):
            with torch.no_grad():
                video = self.generate_base_frames(
                    base_diffusion,
                    height,
                    width,
                    guidance_scale,
                    condition_model_input,
                    negative_condition_model_input,
                )
                if fps in ["medium", "high"]:
                    temporal_groups = self.interpolate_base_frames(
                        base_diffusion,
                        video,
                        height,
                        width,
                        interpolation_guidance_scale,
                        3,
                    )
                    video = self._reshape_temporal_groups(temporal_groups, video)
                    if fps == "high":
                        temporal_groups = self.interpolate_base_frames(
                            base_diffusion,
                            video,
                            height,
                            width,
                            interpolation_guidance_scale,
                            1,
                        )
                        video = self._reshape_temporal_groups(temporal_groups, video)

                self.movq = self.movq_loader()
                vram_info("movq loaded")

                video = torch.cat(
                    [
                        self.movq.decode(frame)
                        for frame in video.chunk(video.shape[0] // 4)
                    ]
                )

                self.movq.to("cpu")
                self.movq = None
                flush()
                vram_info("movq flushed")

                video = torch.clip((video + 1.0) / 2.0, 0.0, 1.0)
                for video_chunk in video.chunk(1):
                    pil_video += [self.to_pil(frame) for frame in video_chunk]
        return pil_video
