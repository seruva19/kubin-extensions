"""
The code has been adopted from fancyfeast/joy-caption-alpha-one
(https://huggingface.co/spaces/fancyfeast/joy-caption-alpha-one/blob/main/app.py)
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
from PIL import Image
import torchvision.transforms.functional as TVF
import json

CHECKPOINT_PATH = "joy-caption-alpha-one"

CLIP_PATH = "google/siglip-so400m-patch14-384"

FORMER_LLM_MODEL_PATH = "meta-llama/Meta-Llama-3.1-8B"
LLM_MODEL_PATH = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"

CHECKPOINT_IMAGE_ADAPTER_URL = "https://huggingface.co/spaces/fancyfeast/joy-caption-alpha-one/resolve/main/9em124t2-499968/image_adapter.pt"
CHECKPOINT_CLIP_MODEL_PATH = "https://huggingface.co/spaces/fancyfeast/joy-caption-alpha-one/resolve/main/9em124t2-499968/clip_model.pt"

CHECKPOINT_VISION_MODEL_URLS = [
    "https://huggingface.co/spaces/fancyfeast/joy-caption-alpha-one/raw/main/9em124t2-499968/text_model/adapter_config.json",
    "https://huggingface.co/spaces/fancyfeast/joy-caption-alpha-one/resolve/main/9em124t2-499968/text_model/adapter_model.safetensors",
]
CHECKPOINT_VISION_MODEL_PATHS = [
    "adapter_config.json",
    "adapter_model.safetensors",
]


caption_type = ["descriptive", "training_prompt", "rng-tags"]
caption_tone = ["formal", "informal"]

caption_length = [
    "any",
    "very short",
    "short",
    "medium-length",
    "long",
    "very long",
] + [str(i) for i in range(20, 261, 10)]


CAPTION_TYPE_MAP = {
    ("descriptive", "formal", False, False): [
        "Write a descriptive caption for this image in a formal tone."
    ],
    ("descriptive", "formal", False, True): [
        "Write a descriptive caption for this image in a formal tone within {word_count} words."
    ],
    ("descriptive", "formal", True, False): [
        "Write a {length} descriptive caption for this image in a formal tone."
    ],
    ("descriptive", "informal", False, False): [
        "Write a descriptive caption for this image in a casual tone."
    ],
    ("descriptive", "informal", False, True): [
        "Write a descriptive caption for this image in a casual tone within {word_count} words."
    ],
    ("descriptive", "informal", True, False): [
        "Write a {length} descriptive caption for this image in a casual tone."
    ],
    ("training_prompt", "formal", False, False): [
        "Write a stable diffusion prompt for this image."
    ],
    ("training_prompt", "formal", False, True): [
        "Write a stable diffusion prompt for this image within {word_count} words."
    ],
    ("training_prompt", "formal", True, False): [
        "Write a {length} stable diffusion prompt for this image."
    ],
    ("rng-tags", "formal", False, False): [
        "Write a list of Booru tags for this image."
    ],
    ("rng-tags", "formal", False, True): [
        "Write a list of Booru tags for this image within {word_count} words."
    ],
    ("rng-tags", "formal", True, False): [
        "Write a {length} list of Booru tags for this image."
    ],
}


class ImageAdapter(nn.Module):
    def __init__(
        self,
        input_features: int,
        output_features: int,
        ln1: bool,
        pos_emb: bool,
        num_image_tokens: int,
        deep_extract: bool,
    ):
        super().__init__()
        self.deep_extract = deep_extract

        if self.deep_extract:
            input_features = input_features * 5

        self.linear1 = nn.Linear(input_features, output_features)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(output_features, output_features)
        self.ln1 = nn.Identity() if not ln1 else nn.LayerNorm(input_features)
        self.pos_emb = (
            None
            if not pos_emb
            else nn.Parameter(torch.zeros(num_image_tokens, input_features))
        )

        self.other_tokens = nn.Embedding(3, output_features)
        self.other_tokens.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, vision_outputs: torch.Tensor):
        if self.deep_extract:
            x = torch.concat(
                (
                    vision_outputs[-2],
                    vision_outputs[3],
                    vision_outputs[7],
                    vision_outputs[13],
                    vision_outputs[20],
                ),
                dim=-1,
            )
            assert len(x.shape) == 3, f"Expected 3, got {len(x.shape)}"
            assert (
                x.shape[-1] == vision_outputs[-2].shape[-1] * 5
            ), f"Expected {vision_outputs[-2].shape[-1] * 5}, got {x.shape[-1]}"
        else:
            x = vision_outputs[-2]

        x = self.ln1(x)

        if self.pos_emb is not None:
            assert (
                x.shape[-2:] == self.pos_emb.shape
            ), f"Expected {self.pos_emb.shape}, got {x.shape[-2:]}"
            x = x + self.pos_emb

        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)

        other_tokens = self.other_tokens(
            torch.tensor([0, 1], device=self.other_tokens.weight.device).expand(
                x.shape[0], -1
            )
        )
        assert other_tokens.shape == (
            x.shape[0],
            2,
            x.shape[2],
        ), f"Expected {(x.shape[0], 2, x.shape[2])}, got {other_tokens.shape}"
        x = torch.cat((other_tokens[:, 0:1], x, other_tokens[:, 1:2]), dim=1)

        return x

    def get_eot_embedding(self):
        return self.other_tokens(
            torch.tensor([2], device=self.other_tokens.weight.device)
        ).squeeze(0)


class JoyCaptionAlphaOneInterrogatorModel:
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
            model_path = os.path.join(cache_dir, CHECKPOINT_PATH)
            if not os.path.exists(model_path):
                os.makedirs(model_path)

            print("loading CLIP")
            self.clip_processor = AutoProcessor.from_pretrained(
                CLIP_PATH, cache_dir=cache_dir
            )

            self.clip_model = AutoModel.from_pretrained(CLIP_PATH, cache_dir=cache_dir)
            self.clip_model = self.clip_model.vision_model

            clip_model_url, clip_download_path = (
                CHECKPOINT_CLIP_MODEL_PATH,
                os.path.join(model_path, "clip_model.pt"),
            )

            if not os.path.exists(clip_download_path):
                print(
                    f"downloading clip model weights from {clip_model_url} to {clip_download_path}"
                )
                urllib.request.urlretrieve(clip_model_url, clip_download_path)
                print("clip model weights downloaded")

            print("loading CLIP model")
            checkpoint = torch.load(clip_download_path, map_location="cpu")
            checkpoint = {
                k.replace("_orig_mod.module.", ""): v for k, v in checkpoint.items()
            }
            self.clip_model.load_state_dict(checkpoint)
            del checkpoint

            self.clip_model.eval()
            self.clip_model.requires_grad_(False)
            self.clip_model.to(self.device)

            print("loading tokenizer")
            self.tokenizer = AutoTokenizer.from_pretrained(
                LLM_MODEL_PATH, use_fast=False, cache_dir=cache_dir
            )

            print("loading LLM")
            vlm_folder = os.path.join(model_path, "text_model")
            if not os.path.exists(vlm_folder):
                os.makedirs(vlm_folder)

            custom_text_model_url, custom_text_model_download_paths = (
                CHECKPOINT_VISION_MODEL_URLS,
                [
                    os.path.join(vlm_folder, path)
                    for path in CHECKPOINT_VISION_MODEL_PATHS
                ],
            )

            if not all(
                os.path.exists(path) for path in custom_text_model_download_paths
            ):
                if not os.path.exists(model_path):
                    os.makedirs(model_path)

                for vlm_file_url, vlm_file_path in zip(
                    custom_text_model_url, custom_text_model_download_paths
                ):
                    print(
                        f"downloading VLM custom text model {vlm_file_url} to {vlm_file_path}"
                    )
                    urllib.request.urlretrieve(vlm_file_url, vlm_file_path)
                    print("VLM custom text models downloaded")

                    config_path = custom_text_model_download_paths[0]
                    with open(config_path, "r") as file:
                        data = json.load(file)
                    data["base_model_name_or_path"] = (
                        "unsloth/Meta-Llama-3.1-8B-bnb-4bit"
                    )
                    with open(config_path, "w") as file:
                        json.dump(data, file, indent=2)
                    print("LLM JSON config was changed")

            self.text_model = AutoModelForCausalLM.from_pretrained(
                vlm_folder,
                device_map=0,
                torch_dtype=torch.bfloat16,
                cache_dir=cache_dir,
            )
            self.text_model.eval()

            print("loading image adapter")
            self.image_adapter = ImageAdapter(
                self.clip_model.config.hidden_size,
                self.text_model.config.hidden_size,
                False,
                False,
                38,
                False,
            )

            image_adapter_model_url, image_adapter_download_path = (
                CHECKPOINT_IMAGE_ADAPTER_URL,
                os.path.join(model_path, "image_adapter.pt"),
            )

            if not os.path.exists(image_adapter_download_path):
                print(
                    f"downloading image adapter weights from {image_adapter_model_url} to {image_adapter_download_path}"
                )
                urllib.request.urlretrieve(
                    image_adapter_model_url, image_adapter_download_path
                )
                print("image adapter weights downloaded")

            self.image_adapter.load_state_dict(
                torch.load(image_adapter_download_path, map_location="cpu")
            )
            self.image_adapter.eval()
            self.image_adapter.to(self.device)

            self.initialized = True

    def get_caption(self, input_image, model_prompt):
        caption_type = model_prompt.split(",")[0]
        caption_tone = model_prompt.split(",")[1]
        caption_length = model_prompt.split(",")[2]

        torch.cuda.empty_cache()
        length = None if caption_length == "any" else caption_length
        if isinstance(length, str):
            try:
                length = int(length)
            except ValueError:
                pass

        if caption_type == "rng-tags" or caption_type == "training_prompt":
            caption_tone = "formal"

        prompt_key = (
            caption_type,
            caption_tone,
            isinstance(length, str),
            isinstance(length, int),
        )
        if prompt_key not in CAPTION_TYPE_MAP:
            raise ValueError(f"Invalid caption type: {prompt_key}")

        prompt_str = CAPTION_TYPE_MAP[prompt_key][0].format(
            length=length, word_count=length
        )
        print(f"joy-caption prompt: {prompt_str}")

        image = input_image.resize((384, 384), Image.LANCZOS)
        pixel_values = TVF.pil_to_tensor(image).unsqueeze(0) / 255.0
        pixel_values = TVF.normalize(pixel_values, [0.5], [0.5])
        pixel_values = pixel_values.to(self.device)

        prompt = self.tokenizer.encode(
            prompt_str,
            return_tensors="pt",
            padding=False,
            truncation=False,
            add_special_tokens=False,
        )

        with torch.amp.autocast_mode.autocast("cuda", enabled=True):
            vision_outputs = self.clip_model(
                pixel_values=pixel_values, output_hidden_states=True
            )
            image_features = vision_outputs.hidden_states
            embedded_images = self.image_adapter(image_features)
            embedded_images = embedded_images.to("cuda")

        prompt_embeds = self.text_model.model.embed_tokens(prompt.to("cuda"))
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
        eot_embed = (
            self.image_adapter.get_eot_embedding()
            .unsqueeze(0)
            .to(dtype=self.text_model.dtype)
        )

        inputs_embeds = torch.cat(
            [
                embedded_bos.expand(embedded_images.shape[0], -1, -1),
                embedded_images.to(dtype=embedded_bos.dtype),
                prompt_embeds.expand(embedded_images.shape[0], -1, -1),
                eot_embed.expand(embedded_images.shape[0], -1, -1),
            ],
            dim=1,
        )

        input_ids = torch.cat(
            [
                torch.tensor([[self.tokenizer.bos_token_id]], dtype=torch.long),
                torch.zeros((1, embedded_images.shape[1]), dtype=torch.long),
                prompt,
                torch.tensor(
                    [[self.tokenizer.convert_tokens_to_ids("<|eot_id|>")]],
                    dtype=torch.long,
                ),
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
            suppress_tokens=None,
        )
        generate_ids = generate_ids[:, input_ids.shape[1] :]
        if generate_ids[0][-1] == self.tokenizer.eos_token_id or generate_ids[0][
            -1
        ] == self.tokenizer.convert_tokens_to_ids("<|eot_id|>"):
            generate_ids = generate_ids[:, :-1]

        caption = self.tokenizer.batch_decode(
            generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )[0]
        return caption.strip()
