# This source code is licensed under the Apache License found in the
# LICENSE file in the current directory.
"""
The code has been adopted from KandinskyVideo
(https://github.com/ai-forever/KandinskyVideo/blob/main/kandinsky_video/condition_encoders.py)
"""


import torch
from torch import nn
from transformers import T5EncoderModel, BitsAndBytesConfig
from typing import Optional, Union

from models.model_31.kandinsky31.utils import freeze


class T5TextConditionEncoder(nn.Module):
    IMAGE_GENERATION_LABEL = "projection_base"
    KEY_FRAME_GENERATION_LABEL = "projection_base"
    INTERPOLATION_LABEL = "projection_interpolation"

    def __init__(
        self,
        model_path,
        t5_projections_path,
        context_dim,
        cache_dir,
        low_cpu_mem_usage: bool = True,
        dtype: Union[str, torch.dtype] = torch.float32,
    ):
        super().__init__()

        t5_model_encoder = T5EncoderModel.from_pretrained(
            model_path,
            low_cpu_mem_usage=low_cpu_mem_usage,
            device_map="auto",
            load_in_4bit=True,
            torch_dtype=torch.bfloat16,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_type=torch.float16,
                bnb_4bit_quant_type="nf4",
            ),
            cache_dir=cache_dir,
        )

        self.encoder = t5_model_encoder.encoder
        self.encoder = freeze(self.encoder)

        self.projection = nn.Sequential(
            nn.Linear(self.encoder.config.d_model, context_dim, bias=False),
            nn.LayerNorm(context_dim),
        )

        self.projection_interpolation = nn.Sequential(
            nn.Linear(self.encoder.config.d_model, context_dim, bias=False),
            nn.LayerNorm(context_dim),
        )

        projections = torch.load(t5_projections_path, map_location="cuda")

        self.projection.load_state_dict(projections[self.IMAGE_GENERATION_LABEL])
        self.projection_interpolation.load_state_dict(
            projections[self.INTERPOLATION_LABEL]
        )

        self.projection.to(device="cuda", dtype=dtype).eval()
        self.projection_interpolation.to(device="cuda", dtype=dtype).eval()

    def forward(self, model_input, model_type):
        embeddings = self.encoder(**model_input).last_hidden_state
        if model_type in [self.IMAGE_GENERATION_LABEL, self.KEY_FRAME_GENERATION_LABEL]:
            context = self.projection(embeddings)
        elif model_type == self.INTERPOLATION_LABEL:
            context = self.projection_interpolation(embeddings)
        else:
            raise Exception(f"Unrecognized projection type {model_type}")

        if "attention_mask" in model_input:
            context_mask = model_input["attention_mask"]
            context[context_mask == 0] = torch.zeros_like(context[context_mask == 0])
            max_seq_length = context_mask.sum(-1).max() + 1
            context = context[:, :max_seq_length]
            context_mask = context_mask[:, :max_seq_length]
        else:
            context_mask = torch.ones(
                *embeddings.shape[:-1], dtype=torch.long, device=embeddings.device
            )
        return context, context_mask


def get_condition_encoder(conf):
    return T5TextConditionEncoder(**conf)
