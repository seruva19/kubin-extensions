import os
import io
import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    BitsAndBytesConfig,
)
from typing import Optional
from decord import cpu, VideoReader, bridge
from qwen_vl_utils import process_vision_info

STEP3_VL_10B_MODEL_ID = "stepfun-ai/Step3-VL-10B"

KEY_MAPPING = {
    "^vision_model": "model.vision_model",
    r"^model(?!\.(language_model|vision_model))": "model.language_model",
    "vit_large_projector": "model.vit_large_projector",
}


def filter_thinking_tags(text: str) -> str:
    import re

    # Also remove any remaining XML-style tags and their content
    # This removes <tag>...</tag>, </tag>, and <tag/> patterns
    # Continues until no more XML tags are found
    while True:
        # Find any XML-style tag
        tag_match = re.search(r"<[^>]+>", text)
        if not tag_match:
            break  # No more tags found

        tag = tag_match.group()

        # Check if it's a closing tag or self-closing tag
        if tag.startswith("</") or tag.endswith("/>"):
            # Remove just the tag
            text = text[: tag_match.start()] + text[tag_match.end() :]
        else:
            # It's an opening tag, find the matching closing tag
            tag_name = re.match(r"<([^>\s/]+)", tag).group(1)
            closing_tag = f"</{tag_name}>"

            # Find the closing tag
            closing_match = re.search(re.escape(closing_tag), text[tag_match.end() :])

            if closing_match:
                # Remove everything from opening tag to closing tag (inclusive)
                end_pos = tag_match.end() + closing_match.end()
                text = text[: tag_match.start()] + text[end_pos:]
            else:
                # No closing tag found, remove just the opening tag
                text = text[: tag_match.start()] + text[tag_match.end() :]

    # Clean up any extra whitespace
    text = text.strip()

    return text


def load_video_frames(video_path: str, frames_per_second: int = 1) -> list:
    from PIL import Image

    bridge.set_bridge("torch")

    with open(video_path, "rb") as f:
        mp4_stream = f.read()
    video_stream = io.BytesIO(mp4_stream)

    decord_vr = VideoReader(video_stream, ctx=cpu(0))

    total_frames = len(decord_vr)
    if total_frames == 0:
        return []

    # Calculate video duration in seconds
    fps = decord_vr.get_avg_fps()
    duration_seconds = total_frames / fps

    # Calculate number of frames to extract (1 per second)
    num_frames = int(duration_seconds * frames_per_second)

    # Ensure at least 1 frame and cap at reasonable maximum
    num_frames = max(1, min(num_frames, 32))

    # Extract frames evenly distributed throughout video
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    video_data = decord_vr.get_batch(frame_indices)
    video_data = video_data.permute(3, 0, 1, 2)

    # Convert to PIL Images
    frames = []
    for i in range(video_data.shape[1]):
        frame = video_data[:, i, :, :]
        frame = frame.permute(1, 2, 0).cpu().numpy()
        frame = Image.fromarray(frame.astype("uint8"))
        frames.append(frame)

    return frames


def init_step3_vl_10b(
    state: dict,
    device: str,
    cache_dir: str,
    quantization: str,
    use_flash_attention: bool = False,
) -> None:
    state["name"] = STEP3_VL_10B_MODEL_ID

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

    # Load processor
    state["processor"] = AutoProcessor.from_pretrained(
        STEP3_VL_10B_MODEL_ID,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )

    # Load model with key mapping for proper model structure
    state["model"] = AutoModelForCausalLM.from_pretrained(
        STEP3_VL_10B_MODEL_ID,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype="auto",
        key_mapping=KEY_MAPPING,
        quantization_config=q_conf,
    ).eval()

    state["device"] = device
    if q_conf is None:
        state["q"] = None
    else:
        state["q"] = q_conf

    def interrogate(
        file_path: str, question: str, use_audio_in_video: bool = False
    ) -> str:
        # Validate file exists
        if not os.path.exists(file_path):
            return f"Error: No video provided - file '{file_path}' does not exist."

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

        try:
            if file_extension in image_extensions:
                # Process as image
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": file_path},
                            {"type": "text", "text": question},
                        ],
                    }
                ]
            else:
                # Process as video by extracting frames
                frames = load_video_frames(file_path, frames_per_second=1)

                if not frames:
                    return f"Error: Could not extract frames from video '{file_path}'"

                # Build content with multiple frames as images
                content = []
                for frame in frames:
                    content.append({"type": "image", "image": frame})
                content.append({"type": "text", "text": question})

                messages = [
                    {
                        "role": "user",
                        "content": content,
                    }
                ]

            # Apply chat template (without tokenizing)
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Process vision inputs (images/videos) using Qwen3-VL utility
            image_inputs, video_inputs = process_vision_info(messages)

            # Prepare model inputs with both text and vision data
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(model.device)

            # Generate response
            with torch.inference_mode():
                generate_ids = model.generate(
                    **inputs, max_new_tokens=2048, do_sample=False
                )

            # Decode output (remove input tokens)
            out_ids_trimmed = [
                o[len(i) :] for i, o in zip(inputs.input_ids, generate_ids)
            ]
            decoded = processor.batch_decode(
                out_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            # Filter out thinking tags and content
            filtered_output = filter_thinking_tags(decoded)

            # Explicit cleanup to free VRAM
            del generate_ids
            del inputs
            if image_inputs:
                del image_inputs
            if video_inputs:
                del video_inputs

            # Force garbage collection and clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return filtered_output

        except Exception as e:
            print(f"Error during Step3-VL-10B inference: {e}")
            import traceback

            traceback.print_exc()

            # Cleanup on error to prevent VRAM leaks
            try:
                if "generate_ids" in locals():
                    del generate_ids
                if "inputs" in locals():
                    del inputs
                if "image_inputs" in locals():
                    del image_inputs
                if "video_inputs" in locals():
                    del video_inputs

                # Force garbage collection and clear CUDA cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass  # Ignore cleanup errors

            return None

    state["fn"] = interrogate
