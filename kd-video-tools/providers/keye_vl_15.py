import os
import torch
from transformers import AutoModel, AutoProcessor, BitsAndBytesConfig
from keye_vl_utils import process_vision_info
import re

KEYE_VL_15_MODEL_ID = "Kwai-Keye/Keye-VL-1.5-8B"


def init_keye_vl_15(state, device, cache_dir, quantization, use_flash_attention):
    state["name"] = KEYE_VL_15_MODEL_ID

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

    if "VIDEO_MAX_PIXELS" not in os.environ:
        os.environ["VIDEO_MAX_PIXELS"] = str(int(32000 * 28 * 28 * 0.9))

    model_load_errors = []
    name_variants = [
        state["name"],
        state["name"].replace(".", "_"),
        state["name"].replace(".", ""),
        state["name"].replace("1.5", "1_5"),
    ]
    state["model"] = None
    for candidate in name_variants:
        try:
            state["model"] = AutoModel.from_pretrained(
                candidate,
                torch_dtype="auto",
                device_map="auto",
                attn_implementation=(
                    "flash_attention_2" if use_flash_attention else "sdpa"
                ),
                quantization_config=q_conf,
                trust_remote_code=True,
                cache_dir=cache_dir,
            )
            if candidate != state["name"]:
                print(
                    f"Loaded Keye model using candidate id '{candidate}' (original: '{state['name']}')"
                )
            break
        except ModuleNotFoundError as e:
            model_load_errors.append((candidate, e))
        except Exception as e:
            print(f"Error loading model candidate '{candidate}': {e}")
            raise

    if state["model"] is None:
        print("Failed to load Keye-VL model. Tried the following id variants:")
        for cand, err in model_load_errors:
            print(f" - {cand}: {err}")
        print(
            "If you see ModuleNotFoundError related to `transformers_modules`, try upgrading `transformers` or use a sanitized model id (e.g. replace '.' with '_')."
        )

        if model_load_errors:
            raise model_load_errors[-1][1]
        else:
            raise RuntimeError("Unknown error while loading Keye-VL model")

    proc_load_errors = []
    state["processor"] = None
    for candidate in name_variants:
        try:
            state["processor"] = AutoProcessor.from_pretrained(
                candidate, trust_remote_code=True, cache_dir=cache_dir
            )
            if candidate != state["name"]:
                print(
                    f"Loaded Keye processor using candidate id '{candidate}' (original: '{state['name']}')"
                )
            break
        except ModuleNotFoundError as e:
            proc_load_errors.append((candidate, e))
        except Exception as e:
            print(f"Error loading processor candidate '{candidate}': {e}")
            raise

    if state["processor"] is None:
        print("Failed to load Keye-VL processor. Tried the following id variants:")
        for cand, err in proc_load_errors:
            print(f" - {cand}: {err}")
        print(
            "If you see ModuleNotFoundError related to `transformers_modules`, try upgrading `transformers` or use a sanitized model id (e.g. replace '.' with '_').\n"
        )
        if proc_load_errors:
            raise proc_load_errors[-1][1]
        else:
            raise RuntimeError("Unknown error while loading Keye-VL processor")

    state["q"] = q_conf if q_conf is not None else None

    def clean_output(text):
        text = re.sub(r"<analysis>.*?</analysis>", "", text, flags=re.DOTALL)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

        answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        if answer_match:
            return answer_match.group(1).strip()

        return text.strip()

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
                        {"type": "image", "image": file_path},
                        {"type": "text", "text": question},
                    ],
                }
            ]
        else:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": file_path, "fps": 2.0},
                        {"type": "text", "text": question},
                    ],
                }
            ]

        try:
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            images, videos, video_kwargs = process_vision_info(
                messages, return_video_kwargs=True
            )

            inputs = processor(
                text=text,
                images=images,
                videos=videos,
                padding=True,
                return_tensors="pt",
                **video_kwargs,
            )

            device = next(model.parameters()).device
            inputs = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }

            with torch.inference_mode():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.95,
                    pad_token_id=processor.tokenizer.eos_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id,
                )

                output_text = processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )

                input_length = len(
                    processor.tokenizer.decode(
                        inputs["input_ids"][0], skip_special_tokens=True
                    )
                )
                generated_text = output_text[0][input_length:].strip()

                cleaned_text = clean_output(generated_text)

                return cleaned_text

        except Exception as e:
            print(f"Error during Keye-VL-1.5 inference: {e}")
            import traceback

            traceback.print_exc()
            return None

    state["fn"] = interrogate
