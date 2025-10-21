import os
import torch

from qwen_omni_utils import process_mm_info

# AVoCaDO HF repo
AVOCADO_MODEL_ID = "AVoCaDO-Captioner/AVoCaDO"


def init_avocado(state, device, cache_dir, quantization, use_flash_attention=True, model_id=AVOCADO_MODEL_ID):
    """Initialize AVoCaDO (Qwen2.5-Omni based) model and processor.

    Provides state['model'], state['processor'], and state['fn'] which matches other kd-video-tools inits.
    """
    try:
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
    except Exception as e:
        raise ImportError(f"Required transformers classes for Qwen2.5-Omni not available: {e}")

    state["device"] = device if device else "cuda"
    state["name"] = model_id

    # Load processor
    processor = Qwen2_5OmniProcessor.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True)

    # Adjust dtype and attention as in other files
    attn_impl = "flash_attention_2" if use_flash_attention else None

    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=attn_impl,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )

    # Some Qwen models provide disable_talker
    try:
        model.disable_talker()
    except Exception:
        pass

    model.eval()

    state["model"] = model
    state["processor"] = processor
    state["tokenizer"] = None
    state["q"] = None

    def interrogate(file_path: str, prompt: str = None, use_audio_in_video: bool = True) -> str:
        model = state["model"]
        processor = state["processor"]

        if not prompt:
            prompt = "Describe this video in detail."

        system_prompt = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech. Please respond in English."

        image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}
        file_extension = os.path.splitext(file_path)[1].lower()

        if file_extension in image_extensions:
            messages = [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {"role": "user", "content": [{"type": "image", "image": file_path}, {"type": "text", "text": prompt}]},
            ]
        else:
            messages = [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {"role": "user", "content": [{"type": "video", "video": file_path}, {"type": "text", "text": prompt}]},
            ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        audios, images, videos = process_mm_info(messages, use_audio_in_video=use_audio_in_video)
        inputs = processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=use_audio_in_video,
        )

        # Move to device/dtype
        try:
            inputs = inputs.to(model.device).to(model.dtype)
        except Exception:
            try:
                inputs = inputs.to(state.get("device", "cuda"))
            except Exception:
                pass

        with torch.no_grad():
            # AVoCaDO example used thinker_max_new_tokens and do_sample=False; keep similar generate kwargs
            gen_kwargs = {"use_audio_in_video": use_audio_in_video, "do_sample": False, "thinker_max_new_tokens": 2048}
            try:
                out = model.generate(**inputs, **gen_kwargs)
            except TypeError:
                # Fallback if model.generate returns (sequences, ...) or different signature
                out = model.generate(**inputs)

        # Decode output similar to other inits
        try:
            # If returned object is a GenerateOutput with sequences attribute
            if hasattr(out, "sequences"):
                text_ids = out.sequences
            else:
                text_ids = out

            # If inputs contains input_ids, strip prompt portion
            prompt_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
            if getattr(text_ids, "ndim", None) is None:
                decoded = processor.batch_decode([text_ids[prompt_len:]], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                answer_text = decoded[0]
            else:
                decoded = processor.batch_decode(text_ids[:, prompt_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                answer_text = decoded[0] if isinstance(decoded, list) else str(decoded)
        except Exception as e:
            answer_text = f"Error decoding AVoCaDO output: {e}"

        return answer_text

    state["fn"] = interrogate
