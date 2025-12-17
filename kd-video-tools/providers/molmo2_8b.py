"""
Provider for Molmo2-8B vision-language model.

This module implements support for the Molmo2-8B model developed by Allen Institute for AI (Ai2),
which supports image, video, and multi-image understanding with grounding capabilities.
"""

import os
import torch
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
)
from typing import Optional

MOLMO2_8B_MODEL_ID = "allenai/Molmo2-8B"


def init_molmo2_8b(
    state: dict,
    device: str,
    cache_dir: str,
    quantization: str,
    use_flash_attention: bool = False,
) -> None:
    """
    Initialize the Molmo2-8B model and processor.

    Args:
        state: Dictionary to store model state (model, processor, etc.)
        device: Device to load the model on (e.g., "cuda")
        cache_dir: Directory to cache model files
        quantization: Quantization type ("none", "int8", or "nf4")
        use_flash_attention: Whether to use Flash Attention 2

    Raises:
        ValueError: If quantization type is invalid
    """
    state["name"] = MOLMO2_8B_MODEL_ID

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

    torch_dtype = torch.float16 if q_conf is None else torch.bfloat16

    # Load processor
    state["processor"] = AutoProcessor.from_pretrained(
        MOLMO2_8B_MODEL_ID,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )

    # Load model
    state["model"] = AutoModelForImageTextToText.from_pretrained(
        MOLMO2_8B_MODEL_ID,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if use_flash_attention else None,
        quantization_config=q_conf,
        cache_dir=cache_dir,
    )

    state["device"] = device
    if q_conf is None:
        state["q"] = None
    else:
        state["q"] = q_conf

    def interrogate(file_path: str, question: str, use_audio_in_video: bool = False) -> str:
        """
        Interrogate a video or image file with a text question.

        Args:
            file_path: Path to the video or image file
            question: Text question/prompt to ask about the media
            use_audio_in_video: Whether to use audio in video (not used by Molmo2)

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
                        {"type": "text", "text": question},
                        {"type": "image", "image": file_path},
                    ],
                }
            ]
        else:
            # Process as video
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "video", "video": file_path},
                    ],
                }
            ]

        # Apply chat template and process inputs
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )

        # Move inputs to model device
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generate response
        with torch.inference_mode():
            generated_ids = model.generate(**inputs, max_new_tokens=2048)

        # Decode output (remove input tokens)
        generated_tokens = generated_ids[0, inputs["input_ids"].size(1) :]
        output_text = processor.tokenizer.decode(
            generated_tokens, skip_special_tokens=True
        )

        return output_text

    state["fn"] = interrogate
