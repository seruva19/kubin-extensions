import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers import BitsAndBytesConfig


VIDEOLLAMA3_MODEL_ID = "DAMO-NLP-SG/VideoLLaMA3-7B"


def init_videollama3(state, device, cache_dir, quantization, use_flash_attention):
    state["name"] = VIDEOLLAMA3_MODEL_ID

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

    state["model"] = AutoModelForCausalLM.from_pretrained(
        VIDEOLLAMA3_MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if use_flash_attention else "sdpa",
        quantization_config=q_conf,
        cache_dir=cache_dir,
    )

    state["processor"] = AutoProcessor.from_pretrained(
        VIDEOLLAMA3_MODEL_ID, trust_remote_code=True, cache_dir=cache_dir
    )

    processor = state["processor"]
    model = state["model"]

    def interrogate(video_path, question):
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": {
                            "video_path": video_path,
                            "fps": 1,
                            "max_frames": 256,
                        },
                    },
                    {"type": "text", "text": question},
                ],
            },
        ]

        inputs = processor(
            conversation=conversation,
            add_system_prompt=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )

        inputs = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

        output_ids = model.generate(**inputs, max_new_tokens=1024)
        response = processor.batch_decode(output_ids, skip_special_tokens=True)[
            0
        ].strip()
        return response

    state["fn"] = interrogate
