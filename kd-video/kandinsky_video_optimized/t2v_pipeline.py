# This source code is licensed under the Apache License found in the
# LICENSE file in the current directory.
"""
The code has been adopted from KandinskyVideo
(https://github.com/ai-forever/KandinskyVideo/blob/main/kandinsky_video/t2v_pipeline.py)
"""

import warnings

from typing import List
import PIL

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from einops import repeat, rearrange
import numpy as np

from .model.unet import UNet
from .movq import MoVQ
from .condition_encoders import T5TextConditionEncoder
from .condition_processors import T5TextConditionProcessor
from .model.diffusion import BaseDiffusion
from .utils import flush, vram_info

SKIP_FRAMES_MEDIUM_FPS = 3
SKIP_FRAMES_HIGH_FPS = 1

MOTION_SCORES = {
    "low": 1,
    "medium": 10,
    "high": 50,
    "extreme": 100,
}


class KandinskyVideoT2VPipeline:

    def __init__(
        self,
        device: str,
        dtype_map: dict,
        unet_loader: UNet,
        unet_null_embedding: torch.Tensor,
        interpolation_unet_loader: UNet,
        interpolation_null_embedding: torch.Tensor,
        t5_processor: T5TextConditionProcessor,
        t5_encoder_loader: T5TextConditionEncoder,
        movq_loader: MoVQ,
        video_movq_loader: MoVQ,
    ):
        self.device = device
        self.dtype_map = dtype_map
        self.to_pil = T.ToPILImage()
        self.image_transform = T.Compose(
            [
                T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                T.ToTensor(),
                T.Lambda(lambda img: 2.0 * img - 1.0),
            ]
        )

        self.unet_null_embedding = unet_null_embedding
        self.unet_loader = unet_loader

        self.interpolation_null_embedding = interpolation_null_embedding
        self.interpolation_unet_loader = interpolation_unet_loader

        self.t5_processor = t5_processor
        self.t5_encoder_loader = t5_encoder_loader

        self.movq_loader = movq_loader
        self.video_movq_loader = video_movq_loader
        self.base_diffusion = BaseDiffusion()

        self.num_frames = 12

    def encode_text(self, prompt, batch_size):
        condition_model_input, _ = self.t5_processor.encode(prompt, None)
        for input_type in condition_model_input:
            condition_model_input[input_type] = (
                condition_model_input[input_type].unsqueeze(0).to(self.device)
            )

        self.t5_encoder = self.t5_encoder_loader()
        vram_info("t5_encoder loaded ('encode_text')")
        model_label = self.t5_encoder.IMAGE_GENERATION_LABEL

        # with torch.cuda.amp.autocast(dtype=self.dtype_map["text_encoder"]):
        with torch.autocast("cuda"):
            context, context_mask = self.t5_encoder(condition_model_input, model_label)

        bs_context = repeat(context, "1 n d -> b n d", b=batch_size)
        bs_context_mask = repeat(context_mask, "1 n -> b n", b=batch_size)

        self.t5_encoder.to("cpu")
        self.t5_encoder = None
        flush()
        vram_info("t5_encoder flushed ('encode_text')")

        return bs_context, bs_context_mask

    def generate_key_frame(self, prompt, height=512, width=512, guidance_scale=3.0):
        with torch.no_grad():
            bs_context, bs_context_mask = self.encode_text(prompt, 1)

            # with torch.cuda.amp.autocast(dtype=self.dtype_map["unet"]):
            with torch.autocast("cuda"):
                self.unet = self.unet_loader()
                vram_info("unet loaded ('generate_key_frame')")

                key_frame = self.base_diffusion.p_sample_loop_image(
                    self.unet,
                    (1, 4, height // 4, width // 4),
                    self.device,
                    bs_context,
                    bs_context_mask,
                    self.unet_null_embedding,
                    guidance_scale,
                )

            return self.decode_image(key_frame)

    def generate_base_frames(
        self,
        prompt,
        key_frame=None,
        height=512,
        width=512,
        guidance_scale_prompt=5,
        guidance_weight_image=3.0,
        motion="normal",
        noise_augmentation=20,
        key_frame_guidance_scale=3.0,
    ):
        if key_frame is None:
            key_frame = self.generate_key_frame(
                prompt, height, width, key_frame_guidance_scale
            )

        if motion not in MOTION_SCORES.keys():
            warnings.warn(
                f"motion must be in {MOTION_SCORES.keys()}. set default speed to medium."
            )
            motion = "medium"
        motion_score = MOTION_SCORES[motion]

        with torch.no_grad():
            bs_context, bs_context_mask = self.encode_text(prompt, self.num_frames)
            key_frame = self.encode_image(key_frame, width, height)

            self.unet = self.unet_loader()
            vram_info("unet loaded ('generate_base_frames')")

            with torch.autocast("cuda"):
                # with torch.cuda.amp.autocast(dtype=self.dtype_map["unet"]):
                base_frames = self.base_diffusion.p_sample_loop(
                    self.unet,
                    (self.num_frames, 4, height // 8, width // 8),
                    "cuda",
                    bs_context,
                    bs_context_mask,
                    self.unet_null_embedding,
                    guidance_scale_prompt,
                    key_frame=key_frame,
                    num_frames=self.num_frames,
                    guidance_weight_image=guidance_weight_image,
                    motion_score=motion_score,
                    noise_augmentation=noise_augmentation,
                )

                self.unet.to("cpu")
                self.unet = None
                flush()
                vram_info("unet flushed")

        return base_frames

    def interpolate_base_frames(
        self, base_frames, height, width, guidance_scale, skip_frames, prompt
    ):

        with torch.no_grad():
            num_predicted_groups = base_frames.shape[0] - 1

            bs_context, bs_context_mask = self.encode_text(prompt, num_predicted_groups)

            with torch.autocast("cuda"):
                # with torch.cuda.amp.autocast(dtype=self.dtype_map["interpolation_unet"]):
                self.interpolation_unet = self.interpolation_unet_loader()
                vram_info("interpolation_unet loaded ('interpolate_base_frames')")

                interpolated_base_frames = (
                    self.base_diffusion.p_sample_loop_interpolation(
                        self.interpolation_unet,
                        (num_predicted_groups, 12, height // 8, width // 8),
                        self.device,
                        base_frames,
                        bs_context,
                        bs_context_mask,
                        self.interpolation_null_embedding,
                        guidance_scale,
                        skip_frames=skip_frames,
                    )
                )

                self.interpolation_unet.to("cpu")
                self.interpolation_unet = None
                flush()
                vram_info("interpolation_unet flushed ('interpolate_base_frames')")

            temporal_groups = rearrange(
                interpolated_base_frames, "b (t c) h w -> b t c h w", t=3
            )
            temporal_groups = torch.cat(
                [temporal_groups[i] for i in range(temporal_groups.shape[0])], axis=0
            )

            b, c, h, w = base_frames.shape
            video_upsampled = torch.zeros(
                (4 * b - 3, c, h, w), device=base_frames.device
            )

            interpolation_indices = [
                i for i in range(video_upsampled.shape[0]) if i % 4 != 0
            ]
            keyframes_indices = [
                i for i in range(video_upsampled.shape[0]) if i % 4 == 0
            ]

            video_upsampled[interpolation_indices] = temporal_groups
            video_upsampled[keyframes_indices] = base_frames

            return video_upsampled

    def encode_image(self, image, width, height):
        with torch.no_grad():
            reduce_factor = max(1, min(image.size[0] / width, image.size[1] / height))
            image = image.resize(
                (
                    round(image.size[0] / reduce_factor),
                    round(image.size[1] / reduce_factor),
                )
            )
            old_width, old_height = image.size
            left = (old_width - width) / 2
            top = (old_height - height) / 2
            right = (old_width + width) / 2
            bottom = (old_height + height) / 2
            image = image.crop((left, top, right, bottom))
            image = self.image_transform(image)

            # with torch.cuda.amp.autocast(dtype=self.dtype_map["movq"]):
            with torch.autocast("cuda"):
                self.movq = self.movq_loader()
                vram_info("movq loaded ('encode_image')")

                image = image.unsqueeze(0).to(
                    device=self.device, dtype=self.dtype_map["movq"]
                )
                image = self.movq.encode(image)[0]

                self.movq.to("cpu")
                self.movq = None
                flush()
                vram_info("movq flushed ('encode_image')")

            return image

    def decode_image(self, image):
        with torch.no_grad():
            with torch.autocast("cuda"):
                # with torch.cuda.amp.autocast(dtype=self.dtype_map["movq"]):
                self.movq = self.movq_loader()
                vram_info("movq loaded ('decode_image')")

                image = self.movq.decode(image)
                image = torch.clip((image + 1.0) / 2.0, 0.0, 1.0)
                image = 255.0 * image.permute(0, 2, 3, 1).cpu().numpy()[0]

                self.movq.to("cpu")
                self.movq = None
                flush()
                vram_info("movq flushed ('decode_image')")

        return PIL.Image.fromarray(image.astype(np.uint8))

    def decode_video(self, video):
        pil_video = []
        with torch.no_grad():
            with torch.autocast("cuda"):
                # with torch.cuda.amp.autocast(dtype=self.dtype_map["movq"]):
                self.video_movq = self.video_movq_loader()
                vram_info("video_movq loaded ('decode_video')")

                video = torch.cat(
                    [
                        self.video_movq.decode(frame)
                        for frame in video.chunk(video.shape[0] // 4)
                    ]
                )
                video = torch.clip((video + 1.0) / 2.0, 0.0, 1.0)
                for video_chunk in video.chunk(1):
                    pil_video += [self.to_pil(frame) for frame in video_chunk]

                self.video_movq.to("cpu")
                self.video_movq = None
                flush()

                vram_info("video_movq flushed ('decode_video')")
        return pil_video

    def __call__(
        self,
        prompt: str,
        negative_prompt: str = None,
        image: PIL.Image.Image = None,
        width: int = 512,
        height: int = 512,
        fps: str = "low",
        motion: str = "normal",
        key_frame_guidance_scale: float = 5.0,
        guidance_weight_prompt: float = 5.0,
        guidance_weight_image: float = 2.0,
        interpolation_guidance_scale: float = 0.5,
        noise_augmentation=20,
    ) -> List[PIL.Image.Image]:

        video = self.generate_base_frames(
            prompt,
            image,
            height,
            width,
            guidance_weight_prompt,
            guidance_weight_image,
            motion,
            noise_augmentation,
            key_frame_guidance_scale,
        )

        if fps in ["medium", "high"]:
            video = self.interpolate_base_frames(
                video,
                height,
                width,
                interpolation_guidance_scale,
                SKIP_FRAMES_MEDIUM_FPS,
                prompt,
            )

            if fps == "high":
                video = self.interpolate_base_frames(
                    video,
                    height,
                    width,
                    interpolation_guidance_scale,
                    SKIP_FRAMES_HIGH_FPS,
                    prompt,
                )

        pil_video = self.decode_video(video)
        return pil_video
