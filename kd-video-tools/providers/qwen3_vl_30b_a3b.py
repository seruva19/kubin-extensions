import os
import torch

try:
    from transformers import (
        Qwen3VLMoeForConditionalGeneration,
        AutoProcessor,
    )
except ImportError:
    print(
        "Qwen3VLMoeForConditionalGeneration cannot be imported. Please upgrade transformers"
    )

import decord
from qwen_vl_utils import process_vision_info
from PIL import Image
from transformers import BitsAndBytesConfig

QWEN3_VL_30B_A3B_MODEL_ID = "Qwen/Qwen3-VL-30B-A3B-Thinking"


def init_qwen3_vl_30b_a3b(state, device, cache_dir, quantization, use_flash_attention):
    state["name"] = QWEN3_VL_30B_A3B_MODEL_ID

    q_conf = None
    if quantization == "none":
        pass
    elif quantization == "nf4":
        q_conf = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_enable_fp32_cpu_offload=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=False,
        )
    elif quantization == "int8":
        q_conf = BitsAndBytesConfig(load_in_8bit=True)

    min_pixels = 256 * 28 * 28
    max_pixels = 1280 * 28 * 28

    state["model"] = Qwen3VLMoeForConditionalGeneration.from_pretrained(
        state["name"],
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if use_flash_attention else "sdpa",
        quantization_config=q_conf,
        device_map="balanced",
        cache_dir=cache_dir,
        trust_remote_code=True,
    )

    state["processor"] = AutoProcessor.from_pretrained(
        state["name"],
        use_fast=True,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )

    if q_conf is None:
        state["model"].to(device=device)
        state["q"] = None
    else:
        state["q"] = q_conf

    def interrogate(file_path, question):
        model = state["model"]
        processor = state["processor"]

        image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}
        file_extension = os.path.splitext(file_path)[1].lower()

        if file_extension in image_extensions:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": file_path,
                        },
                        {"type": "text", "text": question},
                    ],
                }
            ]
        else:
            video_reader = decord.VideoReader(file_path)
            fps = 5  # Default FPS for video processing

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": file_path,
                            "max_pixels": max_pixels,
                            "fps": fps,
                        },
                        {"type": "text", "text": question},
                    ],
                }
            ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        with torch.inference_mode():
            # Generate with increased max_new_tokens for thinking model
            out_ids = model.generate(**inputs, max_new_tokens=2048)
        out_ids_trimmed = [o[len(i) :] for i, o in zip(inputs.input_ids, out_ids)]
        output_text = processor.batch_decode(
            out_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return output_text[0]

    state["fn"] = interrogate
