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

try:
    from transformers import video_utils as _video_utils
except Exception:  # pragma: no cover - fallback for older transformers
    _video_utils = None

try:
    from transformers import utils as _transformers_utils
except Exception:  # pragma: no cover - fallback for older transformers
    _transformers_utils = None

if _video_utils is not None and not hasattr(_video_utils, "make_batched_metadata"):

    def _make_batched_metadata(videos, video_metadata=None):
        # Minimal shim for older transformers versions.
        if video_metadata is None:
            return [None] * len(videos)

        VideoMetadata = getattr(_video_utils, "VideoMetadata", None)

        def _coerce(meta):
            if meta is None:
                return None
            if VideoMetadata is not None and isinstance(meta, VideoMetadata):
                return meta
            if VideoMetadata is not None and isinstance(meta, dict):
                return VideoMetadata(**meta)
            return meta

        if isinstance(video_metadata, (list, tuple)):
            return [_coerce(m) for m in video_metadata]
        return [_coerce(video_metadata)]

    _video_utils.make_batched_metadata = _make_batched_metadata

if _transformers_utils is not None and not hasattr(
    _transformers_utils, "is_torchcodec_available"
):

    def _is_torchcodec_available():
        # Older transformers versions don't expose torchcodec integration.
        return False

    _transformers_utils.is_torchcodec_available = _is_torchcodec_available

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
    # Note: Molmo2's vision backbone has issues with quantization, so we skip it
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
            llm_int8_skip_modules=["vision_backbone", "image_vit", "image_projector"],
        )
    elif quantization == "int8":
        q_conf = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_skip_modules=["vision_backbone", "image_vit", "image_projector"],
        )
    else:
        raise ValueError(f"Unsupported quantization type: {quantization}")

    torch_dtype = torch.float16 if q_conf is None else torch.bfloat16
    state["torch_dtype"] = torch_dtype

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

    def interrogate(
        file_path: str, question: str, use_audio_in_video: bool = False
    ) -> str:
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
        torch_dtype = state.get("torch_dtype", torch.float16)

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

        try:
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
                            {"type": "video", "video": file_path, "max_fps": 8},
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

            # Move inputs to model device and convert dtype
            # Use next(model.parameters()).device for models with device_map="auto"
            device = next(model.parameters()).device

            def process_input_tensor(k, v):
                if not isinstance(v, torch.Tensor):
                    return v

                # Debug: print tensor info
                print(f"Processing input key '{k}': dtype={v.dtype}, shape={v.shape}")

                # Only convert pixel/image data, not integer metadata tensors
                # Check for pixel_values specifically (the actual image/video data)
                is_pixel_data = "pixel_values" in k.lower()

                # Convert pixel data to model dtype
                if is_pixel_data:
                    if v.dtype in [torch.uint8, torch.int8]:
                        # Normalize uint8 [0, 255] to float [0, 1]
                        v = v.float() / 255.0
                    result = v.to(device, dtype=torch_dtype)
                    print(
                        f"  -> Converted to dtype={result.dtype}, device={result.device}"
                    )
                    return result
                # Convert standalone uint8/int8 image tensors (legacy)
                elif v.dtype in [torch.uint8, torch.int8]:
                    if v.dtype == torch.uint8:
                        # Normalize uint8 [0, 255] to float [0, 1]
                        v = v.float() / 255.0
                    return v.to(device, dtype=torch_dtype)
                # Convert float tensors to model dtype
                elif v.dtype in [
                    torch.float16,
                    torch.float32,
                    torch.bfloat16,
                    torch.float64,
                ]:
                    result = v.to(device, dtype=torch_dtype)
                    print(
                        f"  -> Converted to dtype={result.dtype}, device={result.device}"
                    )
                    return result
                # Move other tensors (like input_ids, pooling indices) to device without dtype change
                else:
                    return v.to(device)

            inputs = {k: process_input_tensor(k, v) for k, v in inputs.items()}

            # Generate response
            with torch.inference_mode():
                generated_ids = model.generate(**inputs, max_new_tokens=2048)

            # Decode output (remove input tokens)
            generated_tokens = generated_ids[0, inputs["input_ids"].size(1) :]
            output_text = processor.tokenizer.decode(
                generated_tokens, skip_special_tokens=True
            )

            # Explicit cleanup to free VRAM
            del generated_ids
            del generated_tokens
            del inputs

            # Force garbage collection and clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return output_text

        except Exception as e:
            print(f"Error during Molmo2-8B inference: {e}")
            import traceback

            traceback.print_exc()

            # Cleanup on error to prevent VRAM leaks
            try:
                if "generated_ids" in locals():
                    del generated_ids
                if "generated_tokens" in locals():
                    del generated_tokens
                if "inputs" in locals():
                    del inputs

                # Force garbage collection and clear CUDA cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass  # Ignore cleanup errors

            return None

    state["fn"] = interrogate
