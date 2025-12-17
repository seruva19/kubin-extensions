"""
Provider for NVIDIA-Nemotron-Nano-12B-v2-VL video interrogation model.

This module implements support for the NVIDIA Nemotron Nano 12B v2 VL vision-language model,
which can process images and videos for multimodal understanding tasks.
Supports multi-image reasoning and video understanding with document intelligence capabilities.
"""

import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
)
import decord
from decord import VideoReader, cpu
from PIL import Image
import numpy as np
from typing import Optional, List

NEMOTRON_NANO_12B_V2_VL_MODEL_ID = "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16"


def extract_video_frames(
    video_path: str, fps: float = 1.0, min_frames: int = 8, max_frames: int = 128
) -> List[Image.Image]:
    """
    Extract frames from a video file for Nemotron processing.

    Args:
        video_path: Path to the video file
        fps: Target frames per second to extract (default 1.0)
        min_frames: Minimum number of frames to extract (default 8)
        max_frames: Maximum number of frames to extract (default 128)

    Returns:
        List of PIL Images representing video frames
    """
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        video_fps = vr.get_avg_fps()

        # Calculate number of frames to extract based on target FPS
        # Ensure we stay within min/max bounds
        nframes = int((total_frames / video_fps) * fps)
        nframes = max(min_frames, min(max_frames, nframes))
        nframes = min(nframes, total_frames)

        # Uniformly sample frames
        frame_indices = np.linspace(0, total_frames - 1, nframes, dtype=int)

        # Extract frames and convert to PIL Images
        frames = []
        for idx in frame_indices:
            frame = vr[idx].asnumpy()  # Convert to numpy array (H, W, C)
            # Convert from BGR to RGB if needed, and create PIL Image
            if frame.shape[2] == 3:
                frame_rgb = frame[:, :, ::-1]  # BGR to RGB
            else:
                frame_rgb = frame
            pil_image = Image.fromarray(frame_rgb.astype(np.uint8))
            frames.append(pil_image)

        return frames
    except Exception as e:
        raise RuntimeError(f"Error extracting frames from video {video_path}: {e}")


def init_nemotron_nano_12b_v2_vl(
    state: dict,
    device: str,
    cache_dir: str,
    quantization: str,
    use_flash_attention: bool,
) -> None:
    """
    Initialize the NVIDIA-Nemotron-Nano-12B-v2-VL model and processor.

    Args:
        state: Dictionary to store model state (model, processor, tokenizer, etc.)
        device: Device to load the model on (e.g., "cuda")
        cache_dir: Directory to cache model files
        quantization: Quantization type ("none", "int8", or "nf4")
        use_flash_attention: Whether to use Flash Attention (noted but model may handle this internally)

    Raises:
        ImportError: If required transformers version is not available
        ValueError: If quantization type is invalid
    """
    state["name"] = NEMOTRON_NANO_12B_V2_VL_MODEL_ID

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

    # Load model
    state["model"] = AutoModelForCausalLM.from_pretrained(
        state["name"],
        trust_remote_code=True,
        device_map=device if q_conf is None else "auto",
        torch_dtype=torch.bfloat16,
        quantization_config=q_conf,
        cache_dir=cache_dir,
    ).eval()

    # Load tokenizer and processor
    state["tokenizer"] = AutoTokenizer.from_pretrained(
        state["name"],
        cache_dir=cache_dir,
    )
    state["processor"] = AutoProcessor.from_pretrained(
        state["name"],
        trust_remote_code=True,
        cache_dir=cache_dir,
    )

    # Move model to device if not using quantization
    if q_conf is None:
        state["model"].to(device=device)
        state["q"] = None
    else:
        state["q"] = q_conf

    # Set default video pruning rate for efficient inference
    if hasattr(state["model"], "video_pruning_rate"):
        state["model"].video_pruning_rate = 0.75

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
        tokenizer = state["tokenizer"]

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

        # Prepare messages with /no_think for faster inference
        # Users can modify this to /think for reasoning traces
        if file_extension in image_extensions:
            # Process as image
            image = Image.open(file_path).convert("RGB")

            messages = [
                {"role": "system", "content": "/no_think"},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": ""},
                        {"type": "text", "text": question},
                    ],
                },
            ]

            # Generate prompt and process inputs
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            model_device = next(model.parameters()).device
            inputs = processor(text=[prompt], images=[image], return_tensors="pt").to(
                model_device
            )

            # Generate response
            with torch.inference_mode():
                generated_ids = model.generate(
                    pixel_values=inputs.pixel_values,
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=1024,
                    do_sample=False,
                    eos_token_id=tokenizer.eos_token_id,
                )

            # Extract only the generated tokens (remove input prompt)
            prompt_len = inputs.input_ids.shape[1]
            generated_tokens = generated_ids[0, prompt_len:]

            # Decode output
            output_text = tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            return output_text.strip()

        else:
            # Process as video - extract frames first
            frames = extract_video_frames(
                file_path, fps=1.0, min_frames=8, max_frames=128
            )

            messages = [
                {"role": "system", "content": "/no_think"},
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": ""},
                        {"type": "text", "text": question},
                    ],
                },
            ]

            # Generate prompt
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Process video frames
            model_device = next(model.parameters()).device
            inputs = processor(
                text=[prompt],
                videos=frames,
                return_tensors="pt",
            )
            inputs = inputs.to(model_device)

            # Generate response
            with torch.inference_mode():
                generated_ids = model.generate(
                    pixel_values_videos=inputs.pixel_values_videos,
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=128,
                    do_sample=False,
                    eos_token_id=tokenizer.eos_token_id,
                )

            # Extract only the generated tokens (remove input prompt)
            prompt_len = inputs.input_ids.shape[1]
            generated_tokens = generated_ids[0, prompt_len:]

            # Decode output
            output_text = tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            return output_text.strip()

    state["fn"] = interrogate
