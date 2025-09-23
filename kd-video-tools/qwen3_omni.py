import torch

from qwen_omni_utils import process_mm_info
from transformers import BitsAndBytesConfig

QWEN3_OMNI_MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"


def init_qwen3omni(
    state,
    device,
    cache_dir,
    quantization,
    model_id=QWEN3_OMNI_MODEL_ID,
    use_flash_attention=True,
):
    from transformers import (
        Qwen3OmniMoeForConditionalGeneration,
        Qwen3OmniMoeProcessor,
    )

    state["device"] = "cuda"
    q_conf = BitsAndBytesConfig(load_in_8bit=True) if quantization == "int8" else None

    state["name"] = model_id
    processor = Qwen3OmniMoeProcessor.from_pretrained(model_id, cache_dir=cache_dir)

    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        model_id,
        dtype="auto",
        device_map="auto",
        attn_implementation="flash_attention_2" if use_flash_attention else None,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )
    model.disable_talker()
    model.eval()

    if q_conf is None:
        try:
            model.to(state["device"])
        except Exception:
            pass

    state["model"] = model
    state["processor"] = processor
    state["tokenizer"] = None
    state["q"] = q_conf

    def interrogate(video_path: str, prompt: str) -> str:
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": video_path},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            text = processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            audios, images, videos = process_mm_info(messages, use_audio_in_video=True)

            inputs = processor(
                text=text,
                audio=audios,
                images=images,
                videos=videos,
                return_tensors="pt",
                padding=True,
                use_audio_in_video=True,
            )

            inputs = inputs.to(model.device).to(model.dtype)

            gen_kwargs = {
                "thinker_return_dict_in_generate": True,
                "use_audio_in_video": True,
                "return_audio": False,
            }

            with torch.no_grad():
                text_ids, _ = model.generate(**inputs, **gen_kwargs)

            prompt_len = inputs["input_ids"].shape[1]
            decoded = processor.batch_decode(
                text_ids.sequences[:, prompt_len:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            answer_text = (
                decoded[0] if isinstance(decoded, list) and decoded else str(decoded)
            )

            return answer_text
        except Exception as e:
            return f"Error during Qwen3-Omni interrogation: {e}"

    state["fn"] = interrogate
