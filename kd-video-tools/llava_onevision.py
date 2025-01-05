import os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
import av

from huggingface_hub import snapshot_download

LLAVA_MODEL_ID = "llava-hf/llava-onevision-qwen2-7b-ov-hf"


def init_llava(state, device, cache_dir, quantization, use_flash_attention):
    state["name"] = LLAVA_MODEL_ID

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

    processor = state["processor"] = AutoProcessor.from_pretrained(LLAVA_MODEL_ID)

    model = state["model"] = LlavaOnevisionForConditionalGeneration.from_pretrained(
        LLAVA_MODEL_ID,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        quantization_config=q_conf,
        cache_dir=cache_dir,
        use_flash_attention_2=use_flash_attention and q_conf is None,
    )

    if q_conf is None:
        state["model"].to(device=device)
        state["q"] = None
    else:
        state["q"] = q_conf

    def interrogate(video_path, question):
        container = av.open(video_path)
        total_frames = container.streams.video[0].frames
        indices = np.arange(0, total_frames, total_frames / 8).astype(int)
        video = read_video_pyav(container, indices)

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "video"},
                    {"type": "text", "text": question},
                ],
            },
        ]

        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(videos=list(video), text=prompt, return_tensors="pt").to(
            device, torch.float16
        )

        out = model.generate(**inputs, max_new_tokens=512)
        response = (
            processor.batch_decode(
                out, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )[0]
            .split("assistant\n")[1]
            .strip()
        )
        return response

    state["fn"] = interrogate


def read_video_pyav(container, indices):
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])
