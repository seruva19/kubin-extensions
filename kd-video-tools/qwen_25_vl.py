import os
import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor,
)
import decord
from qwen_vl_utils import process_vision_info
from PIL import Image
from transformers import AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from moviepy.editor import VideoFileClip

QWEN25_VL_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
SKY_CAPTIONER_MODEL_ID = "Skywork/SkyCaptioner-V1"
SHOTVL_MODEL_ID = "Vchitect/ShotVL-7B"


def init_qwen25vl(
    state, device, cache_dir, quantization, model_id, use_flash_attention
):
    state["name"] = model_id

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

    state["model"] = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        state["name"],
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if use_flash_attention else "sdpa",
        quantization_config=q_conf,
        device_map="balanced",
        cache_dir=cache_dir,
    )

    state["processor"] = AutoProcessor.from_pretrained(
        state["name"],
        # revision="refs/pr/24" if state["name"] == SHOTVL_MODEL_ID else None,
        use_fast=True,
        # min_pixels=min_pixels,
        # max_pixels=max_pixels,
        # max_pixels=360*640,
        # fps=12.0,
        cache_dir=cache_dir,
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
            fps = 12 if model_id == SHOTVL_MODEL_ID else 5  # video_reader.get_avg_fps()

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
            out_ids = model.generate(**inputs, max_new_tokens=512)
        out_ids_trimmed = [o[len(i) :] for i, o in zip(inputs.input_ids, out_ids)]
        output_text = processor.batch_decode(
            out_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return output_text[0]

    state["fn"] = interrogate
