"""
Provider for Qwen3-VL-8B-Instruct video interrogation model.

This module implements support for the Qwen3-VL-8B-Instruct vision-language model,
which can process both images and videos for multimodal understanding tasks.
"""

import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    BitsAndBytesConfig,
)
import decord
from qwen_vl_utils import process_vision_info
from PIL import Image
from typing import Optional

QWEN3_VL_8B_INSTRUCT_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"


def init_qwen3_vl_8b_instruct(
    state: dict,
    device: str,
    cache_dir: str,
    quantization: str,
    use_flash_attention: bool,
) -> None:
    """
    Initialize the Qwen3-VL-8B-Instruct model and processor.

    Args:
        state: Dictionary to store model state (model, processor, etc.)
        device: Device to load the model on (e.g., "cuda")
        cache_dir: Directory to cache model files
        quantization: Quantization type ("none", "int8", or "nf4")
        use_flash_attention: Whether to use Flash Attention 2

    Raises:
        ImportError: If required transformers version is not available
        ValueError: If quantization type is invalid
    """
    state["name"] = QWEN3_VL_8B_INSTRUCT_MODEL_ID

    # Configure quantization
    q_conf: Optional[BitsAndBytesConfig] = None
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
    else:
        raise ValueError(f"Unsupported quantization type: {quantization}")

    # Video processing parameters
    min_pixels = 256 * 28 * 28
    max_pixels = 1280 * 28 * 28

    # Load model - try Qwen3VLForConditionalGeneration first, fallback to AutoModelForCausalLM
    try:
        from transformers import Qwen3VLForConditionalGeneration

        model_class = Qwen3VLForConditionalGeneration
    except ImportError:
        # Fallback to AutoModelForCausalLM if specific class not available
        model_class = AutoModelForCausalLM

    state["model"] = model_class.from_pretrained(
        state["name"],
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if use_flash_attention else "sdpa",
        quantization_config=q_conf,
        device_map="balanced",
        cache_dir=cache_dir,
        trust_remote_code=True,
    )

    # Load processor
    state["processor"] = AutoProcessor.from_pretrained(
        state["name"],
        use_fast=True,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )

    # Move model to device if not using quantization
    if q_conf is None:
        state["model"].to(device=device)
        state["q"] = None
    else:
        state["q"] = q_conf

    def interrogate(file_path: str, question: str) -> str:
        """
        Interrogate a video or image file with a text question.

        Args:
            file_path: Path to the video or image file
            question: Text question/prompt to ask about the media

        Returns:
            Generated text response from the model
        """
        model = state["model"]
        processor = state["processor"]

        # Determine if input is image or video
        image_extensions = {
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".bmp",
            ".tiff",
            ".webp",
        }
        file_extension = os.path.splitext(file_path)[1].lower()

        if file_extension in image_extensions:
            # Process as image
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
            # Process as video
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

        # Apply chat template
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Process vision inputs (images/videos)
        image_inputs, video_inputs = process_vision_info(messages)

        # Prepare model inputs
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Generate response
        with torch.inference_mode():
            out_ids = model.generate(**inputs, max_new_tokens=2048)

        # Decode output (remove input tokens)
        out_ids_trimmed = [o[len(i) :] for i, o in zip(inputs.input_ids, out_ids)]
        output_text = processor.batch_decode(
            out_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return output_text[0]

    state["fn"] = interrogate
