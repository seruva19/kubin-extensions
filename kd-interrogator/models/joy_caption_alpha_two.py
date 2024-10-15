"""
The code has been adopted from fancyfeast/joy-caption-alpha-two
(https://huggingface.co/spaces/fancyfeast/joy-caption-alpha-two/blob/main/app.py)
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

CHECKPOINT_PATH = "joy-caption-alpha-two"

CLIP_PATH = "google/siglip-so400m-patch14-384"

FORMER_LLM_MODEL_PATH = "meta-llama/Meta-Llama-3.1-8B"
LLM_MODEL_PATH = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"

CHECKPOINT_IMAGE_ADAPTER_URL = "https://huggingface.co/spaces/fancyfeast/joy-caption-alpha-two/resolve/main/cgrkzexw-599808/image_adapter.pt"
CHECKPOINT_CLIP_MODEL_PATH = "https://huggingface.co/spaces/fancyfeast/joy-caption-alpha-two/resolve/main/cgrkzexw-599808/clip_model.pt"

CHECKPOINT_VISION_MODEL_URLS = [
    "https://huggingface.co/spaces/fancyfeast/joy-caption-alpha-two/resolve/main/cgrkzexw-599808/text_model/adapter_config.json",
    "https://huggingface.co/spaces/fancyfeast/joy-caption-alpha-two/resolve/main/cgrkzexw-599808/text_model/adapter_model.safetensors",
    "https://huggingface.co/spaces/fancyfeast/joy-caption-alpha-two/resolve/main/cgrkzexw-599808/text_model/special_tokens_map.json",
    "https://huggingface.co/spaces/fancyfeast/joy-caption-alpha-two/resolve/main/cgrkzexw-599808/text_model/tokenizer.json",
    "https://huggingface.co/spaces/fancyfeast/joy-caption-alpha-two/resolve/main/cgrkzexw-599808/text_model/tokenizer_config.json",
]
CHECKPOINT_VISION_MODEL_PATHS = [
    "adapter_config.json",
    "adapter_model.safetensors",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
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
    "Descriptive": [
        "Write a descriptive caption for this image in a formal tone.",
        "Write a descriptive caption for this image in a formal tone within {word_count} words.",
        "Write a {length} descriptive caption for this image in a formal tone.",
    ],
    "Descriptive (Informal)": [
        "Write a descriptive caption for this image in a casual tone.",
        "Write a descriptive caption for this image in a casual tone within {word_count} words.",
        "Write a {length} descriptive caption for this image in a casual tone.",
    ],
    "Training Prompt": [
        "Write a stable diffusion prompt for this image.",
        "Write a stable diffusion prompt for this image within {word_count} words.",
        "Write a {length} stable diffusion prompt for this image.",
    ],
    "MidJourney": [
        "Write a MidJourney prompt for this image.",
        "Write a MidJourney prompt for this image within {word_count} words.",
        "Write a {length} MidJourney prompt for this image.",
    ],
    "Booru tag list": [
        "Write a list of Booru tags for this image.",
        "Write a list of Booru tags for this image within {word_count} words.",
        "Write a {length} list of Booru tags for this image.",
    ],
    "Booru-like tag list": [
        "Write a list of Booru-like tags for this image.",
        "Write a list of Booru-like tags for this image within {word_count} words.",
        "Write a {length} list of Booru-like tags for this image.",
    ],
    "Art Critic": [
        "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc.",
        "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it within {word_count} words.",
        "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it {length}.",
    ],
    "Product Listing": [
        "Write a caption for this image as though it were a product listing.",
        "Write a caption for this image as though it were a product listing. Keep it under {word_count} words.",
        "Write a {length} caption for this image as though it were a product listing.",
    ],
    "Social Media Post": [
        "Write a caption for this image as if it were being used for a social media post.",
        "Write a caption for this image as if it were being used for a social media post. Limit the caption to {word_count} words.",
        "Write a {length} caption for this image as if it were being used for a social media post.",
    ],
}

CHOICES = [
    "If there is a person/character in the image you must refer to them as {name}.",
    "Do NOT include information about people/characters that cannot be changed (like ethnicity, gender, etc), but do still include changeable attributes (like hair style).",
    "Include information about lighting.",
    "Include information about camera angle.",
    "Include information about whether there is a watermark or not.",
    "Include information about whether there are JPEG artifacts or not.",
    "If it is a photo you MUST include information about what camera was likely used and details such as aperture, shutter speed, ISO, etc.",
    "Do NOT include anything sexual; keep it PG.",
    "Do NOT mention the image's resolution.",
    "You MUST include information about the subjective aesthetic quality of the image from low to very high.",
    "Include information on the image's composition style, such as leading lines, rule of thirds, or symmetry.",
    "Do NOT mention any text that is in the image.",
    "Specify the depth of field and whether the background is in focus or blurred.",
    "If applicable, mention the likely use of artificial or natural lighting sources.",
    "Do NOT use any ambiguous language.",
    "Include whether the image is sfw, suggestive, or nsfw.",
    "ONLY describe the most important elements of the image.",
]


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


class JoyCaptionAlphaTwoInterrogatorModel:
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
                    if not os.path.exists(vlm_file_path):
                        print(
                            f"downloading VLM custom text model {vlm_file_url} to {vlm_file_path}"
                        )
                        urllib.request.urlretrieve(vlm_file_url, vlm_file_path)
                        print("VLM custom text models downloaded")

                    config_path = custom_text_model_download_paths[0]
                    with open(config_path, "r") as file:
                        data = json.load(file)
                    data["base_model_name_or_path"] = LLM_MODEL_PATH
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

    def get_caption(
        self,
        input_image,
        model_prompt,
        extra_options: list[str] = [],
        name_input: str = "",
        custom_prompt: str = "",
    ):
        caption_type = model_prompt.split(",")[0]
        caption_length = model_prompt.split(",")[1]

        torch.cuda.empty_cache()
        length = None if caption_length == "any" else caption_length
        if isinstance(length, str):
            try:
                length = int(length)
            except ValueError:
                pass

        if length is None:
            map_idx = 0
        elif isinstance(length, int):
            map_idx = 1
        elif isinstance(length, str):
            map_idx = 2
        else:
            raise ValueError(f"Invalid caption length: {length}")

        prompt_str = CAPTION_TYPE_MAP[caption_type][map_idx]
        if len(extra_options) > 0:
            prompt_str += " " + " ".join(extra_options)
        prompt_str = prompt_str.format(
            name=name_input, length=caption_length, word_count=caption_length
        )

        if custom_prompt.strip() != "":
            prompt_str = custom_prompt.strip()
        print(f"joy-caption prompt: {prompt_str}")

        image = input_image.resize((384, 384), Image.LANCZOS)
        pixel_values = TVF.pil_to_tensor(image).unsqueeze(0) / 255.0
        pixel_values = TVF.normalize(pixel_values, [0.5], [0.5])
        pixel_values = pixel_values.to(self.device)

        with torch.amp.autocast_mode.autocast(self.device, enabled=True):
            vision_outputs = self.clip_model(
                pixel_values=pixel_values, output_hidden_states=True
            )
            embedded_images = self.image_adapter(vision_outputs.hidden_states)
            embedded_images = embedded_images.to(self.device)

        conversation = [
            {
                "role": "system",
                "content": "You are a helpful image captioner.",
            },
            {
                "role": "user",
                "content": prompt_str,
            },
        ]

        conversation_string = self.tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        conversation_tokens = self.tokenizer.encode(
            conversation_string,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=False,
        )
        prompt_tokens = self.tokenizer.encode(
            prompt_str, return_tensors="pt", add_special_tokens=False, truncation=False
        )
        conversation_tokens = conversation_tokens.squeeze(0)
        prompt_tokens = prompt_tokens.squeeze(0)
        eot_id_indices = (
            (conversation_tokens == self.tokenizer.convert_tokens_to_ids("<|eot_id|>"))
            .nonzero(as_tuple=True)[0]
            .tolist()
        )
        preamble_len = eot_id_indices[1] - prompt_tokens.shape[0]
        conversation_embeds = self.text_model.model.embed_tokens(
            conversation_tokens.unsqueeze(0).to(self.device)
        )

        input_embeds = torch.cat(
            [
                conversation_embeds[:, :preamble_len],
                embedded_images.to(dtype=conversation_embeds.dtype),
                conversation_embeds[:, preamble_len:],
            ],
            dim=1,
        ).to(self.device)

        input_ids = torch.cat(
            [
                conversation_tokens[:preamble_len].unsqueeze(0),
                torch.zeros((1, embedded_images.shape[1]), dtype=torch.long),
                conversation_tokens[preamble_len:].unsqueeze(0),
            ],
            dim=1,
        ).to(self.device)

        attention_mask = torch.ones_like(input_ids)

        generate_ids = self.text_model.generate(
            input_ids,
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            max_new_tokens=300,
            temperature=0.6,
            top_p=0.9,
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
