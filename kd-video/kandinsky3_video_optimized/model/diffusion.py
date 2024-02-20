# This source code is licensed under the Apache License found in the
# LICENSE file in the current directory.
"""
The code has been adopted from KandinskyVideo
(https://github.com/ai-forever/KandinskyVideo/blob/main/video_kandinsky3/model/diffusion.py)
"""

import math

import torch
from einops import rearrange
from tqdm import tqdm
import pdb

from .utils import get_tensor_items


def get_named_beta_schedule(schedule_name, timesteps):
    if schedule_name == "linear":
        scale = 1000 / timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
    elif schedule_name == "cosine":
        alpha_bar = lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
        betas = []
        for i in range(timesteps):
            t1 = i / timesteps
            t2 = (i + 1) / timesteps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), 0.999))
        return torch.tensor(betas, dtype=torch.float32)


class BaseDiffusion:

    def __init__(self, betas, percentile=None, gen_noise=torch.randn_like):
        self.betas = betas
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat(
            [torch.ones(1, dtype=betas.dtype), self.alphas_cumprod[:-1]]
        )

        # calculate q(x_t | x_{t-1})
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # calculate q(x_{t-1} | x_t, x_0)
        self.posterior_mean_coef_1 = (
            torch.sqrt(self.alphas_cumprod_prev) * betas / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef_2 = (
            torch.sqrt(alphas)
            * (1.0 - self.alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod)
        )
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance = torch.log(
            torch.cat(
                [self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]]
            )
        )

        self.percentile = percentile
        self.time_scale = 1000 // self.num_timesteps
        self.gen_noise = gen_noise

    def process_x_start(self, x_start):
        bs, ndims = x_start.shape[0], len(x_start.shape[1:])
        if self.percentile is not None:
            quantile = torch.quantile(
                rearrange(x_start, "b ... -> b (...)").abs(), self.percentile, dim=-1
            )
            quantile = torch.clip(quantile, min=1.0)
            quantile = quantile.reshape(bs, *((1,) * ndims))
            return torch.clip(x_start, -quantile, quantile) / quantile
        else:
            return torch.clip(x_start, -1.0, 1.0)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = self.gen_noise(x_start)
        sqrt_alphas_cumprod = get_tensor_items(
            self.sqrt_alphas_cumprod, t, x_start.shape
        )
        sqrt_one_minus_alphas_cumprod = get_tensor_items(
            self.sqrt_one_minus_alphas_cumprod, t, noise.shape
        )
        x_t = sqrt_alphas_cumprod * x_start + sqrt_one_minus_alphas_cumprod * noise
        return x_t

    def q_posterior_mean_variance(self, x_start, x_t, t):
        posterior_mean_coef_1 = get_tensor_items(
            self.posterior_mean_coef_1, t, x_start.shape
        )
        posterior_mean_coef_2 = get_tensor_items(
            self.posterior_mean_coef_2, t, x_t.shape
        )
        posterior_mean = posterior_mean_coef_1 * x_start + posterior_mean_coef_2 * x_t

        posterior_variance = get_tensor_items(self.posterior_variance, t, x_start.shape)
        posterior_log_variance = get_tensor_items(
            self.posterior_log_variance, t, x_start.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance

    def text_guidance(
        self,
        model,
        x,
        t,
        context,
        context_mask,
        null_embedding,
        guidance_weight_text,
        temporal_positions=None,
        uncondition_context=None,
        uncondition_context_mask=None,
        base_frames=None,
        num_temporal_groups=None,
        skip_frames=None,
    ):
        large_x = x.repeat(2, 1, 1, 1)
        large_t = t.repeat(2)
        if base_frames is not None:
            left_base_frames, right_base_frames = base_frames
            null_base_frames = torch.zeros_like(left_base_frames)
            large_left_base_frames = torch.cat([left_base_frames, null_base_frames])
            large_right_base_frames = torch.cat([right_base_frames, null_base_frames])
            large_x = torch.cat(
                [large_left_base_frames, large_x, large_right_base_frames], dim=1
            )

        if uncondition_context is None:
            uncondition_context = torch.zeros_like(context)
            uncondition_context_mask = torch.zeros_like(context_mask)
            uncondition_context[:, 0] = null_embedding
            uncondition_context_mask[:, 0] = 1
        large_context = torch.cat([context, uncondition_context])
        large_context_mask = torch.cat([context_mask, uncondition_context_mask])

        if temporal_positions is not None:
            temporal_positions = torch.cat([temporal_positions, temporal_positions])
        if skip_frames is not None:
            skip_frames = torch.cat([skip_frames, skip_frames])

        pred_large_noise = model(
            large_x,
            large_t * self.time_scale,
            large_context,
            large_context_mask.bool(),
            temporal_positions,
            skip_frames,
            num_temporal_groups,
        )
        pred_noise, uncond_pred_noise = torch.chunk(pred_large_noise, 2)
        pred_noise = (
            guidance_weight_text + 1.0
        ) * pred_noise - guidance_weight_text * uncond_pred_noise
        return pred_noise

    def p_mean_variance(
        self,
        model,
        x,
        t,
        context,
        context_mask,
        null_embedding,
        guidance_weight_text,
        temporal_positions=None,
        negative_context=None,
        negative_context_mask=None,
        base_frames=None,
        num_temporal_groups=None,
        skip_frames=None,
        v_predication=False,
    ):

        pred_noise = self.text_guidance(
            model,
            x,
            t,
            context,
            context_mask,
            null_embedding,
            guidance_weight_text,
            temporal_positions=temporal_positions,
            uncondition_context=negative_context,
            uncondition_context_mask=negative_context_mask,
            base_frames=base_frames,
            num_temporal_groups=num_temporal_groups,
            skip_frames=skip_frames,
        )

        sqrt_one_minus_alphas_cumprod = get_tensor_items(
            self.sqrt_one_minus_alphas_cumprod, t, pred_noise.shape
        )
        sqrt_alphas_cumprod = get_tensor_items(
            self.sqrt_alphas_cumprod, t, pred_noise.shape
        )
        if v_predication:
            pred_v = pred_noise
            pred_x_start = (
                sqrt_alphas_cumprod * x - sqrt_one_minus_alphas_cumprod * pred_v
            )
        else:
            pred_x_start = (
                x - sqrt_one_minus_alphas_cumprod * pred_noise
            ) / sqrt_alphas_cumprod
        pred_x_start = self.process_x_start(pred_x_start)

        pred_mean, pred_var, pred_log_var = self.q_posterior_mean_variance(
            pred_x_start, x, t
        )
        return pred_mean, pred_var, pred_log_var

    @torch.no_grad()
    def p_sample(
        self,
        model,
        x,
        t,
        context,
        context_mask,
        null_embedding,
        guidance_weight_text,
        temporal_positions=None,
        negative_context=None,
        negative_context_mask=None,
        base_frames=None,
        num_temporal_groups=None,
        skip_frames=None,
        v_predication=False,
    ):
        bs = x.shape[0]
        ndims = len(x.shape[1:])
        pred_mean, _, pred_log_var = self.p_mean_variance(
            model,
            x,
            t,
            context,
            context_mask,
            null_embedding,
            guidance_weight_text,
            temporal_positions=temporal_positions,
            negative_context=negative_context,
            negative_context_mask=negative_context_mask,
            base_frames=base_frames,
            num_temporal_groups=num_temporal_groups,
            skip_frames=skip_frames,
            v_predication=v_predication,
        )
        noise = torch.randn_like(x)
        mask = (t != 0).reshape(bs, *((1,) * ndims))
        sample = pred_mean + mask * torch.exp(0.5 * pred_log_var) * noise
        return sample

    @torch.no_grad()
    def p_sample_loop(
        self,
        model,
        shape,
        device,
        context,
        context_mask,
        null_embedding,
        guidance_weight_text,
        temporal_positions=None,
        negative_context=None,
        negative_context_mask=None,
        base_frames=None,
        num_temporal_groups=None,
        skip_frames=None,
        v_predication=False,
    ):
        img = torch.randn(*shape, device=device)
        t_start = self.num_timesteps
        time = list(range(t_start))[::-1]
        for time in tqdm(time, position=0):
            time = torch.tensor([time] * shape[0], device=device)
            img = self.p_sample(
                model,
                img,
                time,
                context,
                context_mask,
                null_embedding,
                guidance_weight_text,
                temporal_positions=temporal_positions,
                negative_context=negative_context,
                negative_context_mask=negative_context_mask,
                base_frames=base_frames,
                num_temporal_groups=num_temporal_groups,
                skip_frames=skip_frames,
                v_predication=v_predication,
            )
        return img


def get_diffusion(conf):
    betas = get_named_beta_schedule(**conf.schedule_params)
    base_diffusion = BaseDiffusion(betas, **conf.diffusion_params)
    return base_diffusion
