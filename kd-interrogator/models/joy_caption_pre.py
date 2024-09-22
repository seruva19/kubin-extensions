"""
The code has been adopted from fancyfeast/joy-caption-pre-alpha
(https://huggingface.co/spaces/fancyfeast/joy-caption-pre-alpha/blob/main/app.py)
"""

from torch import nn
import torch
import torch.amp.autocast_mode
import os
import urllib.request
from transformers import (
    AutoModel,
    AutoProcessor,
    AutoTokenizer,
    AutoModelForCausalLM,
)

CLIP_PATH = "google/siglip-so400m-patch14-384"
MODEL_PATH = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"
CHECKPOINT_PATH = "https://huggingface.co/spaces/fancyfeast/joy-caption-pre-alpha/resolve/main/wpkklhc6/image_adapter.pt"


class ImageAdapter(nn.Module):
    def __init__(self, input_features: int, output_features: int):
        super().__init__()
        self.linear1 = nn.Linear(input_features, output_features)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(output_features, output_features)

    def forward(self, vision_outputs: torch.Tensor):
        x = self.linear1(vision_outputs)
        x = self.activation(x)
        x = self.linear2(x)
        return x


class JoyCaptionPreAlphaInterrogatorModel:
    def __init__(self):
        self.initialized = False
        self.device = "cuda"

        self.clip_processor = None
        self.tokenizer = None
        self.clip_model = None
        self.image_adapter = None
        self.text_model = None

    def load_components(self, cache_dir, device):
        self.device = device

        if not self.initialized:
            print("loading CLIP")

            self.clip_processor = AutoProcessor.from_pretrained(
                CLIP_PATH, cache_dir=cache_dir
            )

            self.clip_model = AutoModel.from_pretrained(CLIP_PATH, cache_dir=cache_dir)
            self.clip_model = self.clip_model.vision_model
            self.clip_model.eval()
            self.clip_model.requires_grad_(False)
            self.clip_model.to(self.device)

            print("loading tokenizer")
            self.tokenizer = AutoTokenizer.from_pretrained(
                MODEL_PATH, use_fast=False, cache_dir=cache_dir
            )

            print("loading LLM")
            self.text_model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                cache_dir=cache_dir,
            )
            self.text_model.eval()

            self.image_adapter = ImageAdapter(
                self.clip_model.config.hidden_size, self.text_model.config.hidden_size
            )

            print("loading image adapter")
            model_url, download_path = (
                CHECKPOINT_PATH,
                f"{cache_dir}/joycaption_image_adapter.pt",
            )

            if not os.path.exists(download_path):
                print(
                    f"downloading image adapter weights from {model_url} to {download_path}"
                )
                urllib.request.urlretrieve(model_url, download_path)
                print("image adapter weights downloaded")

            self.image_adapter.load_state_dict(
                torch.load(download_path, map_location="cpu")
            )
            self.image_adapter.eval()
            self.image_adapter.to(self.device)
            self.initialized = True

    def get_caption(
        self,
        input_image,
        model_prompt,
    ):
        image = self.clip_processor(
            images=input_image, return_tensors="pt"
        ).pixel_values
        image = image.to(self.device)

        prompt = self.tokenizer.encode(
            model_prompt,
            return_tensors="pt",
            padding=False,
            truncation=False,
            add_special_tokens=False,
        )

        with torch.amp.autocast_mode.autocast(self.device, enabled=True):
            vision_outputs = self.clip_model(
                pixel_values=image, output_hidden_states=True
            )
            image_features = vision_outputs.hidden_states[-2]
            embedded_images = self.image_adapter(image_features)
            embedded_images = embedded_images.to(self.device)

        prompt_embeds = self.text_model.model.embed_tokens(prompt.to(self.device))
        assert prompt_embeds.shape == (
            1,
            prompt.shape[1],
            self.text_model.config.hidden_size,
        ), f"Prompt shape is {prompt_embeds.shape}, expected {(1, prompt.shape[1], self.text_model.config.hidden_size)}"
        embedded_bos = self.text_model.model.embed_tokens(
            torch.tensor(
                [[self.tokenizer.bos_token_id]],
                device=self.text_model.device,
                dtype=torch.int64,
            )
        )

        inputs_embeds = torch.cat(
            [
                embedded_bos.expand(embedded_images.shape[0], -1, -1),
                embedded_images.to(dtype=embedded_bos.dtype),
                prompt_embeds.expand(embedded_images.shape[0], -1, -1),
            ],
            dim=1,
        )

        input_ids = torch.cat(
            [
                torch.tensor([[self.tokenizer.bos_token_id]], dtype=torch.long),
                torch.zeros((1, embedded_images.shape[1]), dtype=torch.long),
                prompt,
            ],
            dim=1,
        ).to(self.device)
        attention_mask = torch.ones_like(input_ids)

        generate_ids = self.text_model.generate(
            input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=300,
            do_sample=True,
            top_k=10,
            temperature=0.5,
            suppress_tokens=None,
        )

        generate_ids = generate_ids[:, input_ids.shape[1] :]
        if generate_ids[0][-1] == self.tokenizer.eos_token_id:
            generate_ids = generate_ids[:, :-1]

        caption = self.tokenizer.batch_decode(
            generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )[0]
        return caption.strip()
