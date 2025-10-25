import os
import re
import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor,
    BitsAndBytesConfig,
)
from vision_process import process_vision_info

OPEN_O3_MODEL_ID = "marinero4972/Open-o3-Video"


def init_open_o3_video(
    state, device, cache_dir, quantization, use_flash_attention=False
):
    state["name"] = OPEN_O3_MODEL_ID

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

    torch_dtype = torch.float16 if q_conf is None else torch.bfloat16

    state["model"] = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        OPEN_O3_MODEL_ID,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if use_flash_attention else None,
        quantization_config=q_conf,
        cache_dir=cache_dir,
    )

    state["processor"] = AutoProcessor.from_pretrained(
        OPEN_O3_MODEL_ID,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )
    state["processor"].image_processor.image_processor_type = "Qwen2_5_VLImageProcessor"
    state["tokenizer"] = AutoTokenizer.from_pretrained(
        OPEN_O3_MODEL_ID,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )

    state["tokenizer"].padding_side = "left"
    state["processor"].tokenizer = state["tokenizer"]

    if q_conf is None:
        state["q"] = None
    else:
        state["q"] = q_conf

    state["device"] = device

    # system_message = (
    #     "A conversation between user and assistant. The user provides a video and asks a question, "
    #     "and the Assistant solves it. The assistant MUST first think about the reasoning process in the mind "
    #     "and then provide the user with the answer. The reasoning process and answer are enclosed within "
    #     "<think> </think> and <answer> </answer> tags, respectively. All reasoning must be grounded in visual "
    #     "evidence from the video. When you mention any related object, person, or specific visual element in the "
    #     "reasoning process, you must strictly follow the following format: "
    #     "`<obj>object_name</obj><box>bounding_box</box>at<t>time_in_seconds</t>s`. "
    #     "The answer part only requires a text response; tags like <obj>, <box>, <t> are not needed."
    # )

    system_message = (
        "You are Open-o3-Video, a large multimodal model. "
        "You can understand videos and answer questions about them in detail. "
        "When you provide answers, ensure they are enclosed within <answer> </answer> tags."
    )

    def interrogate(video_path, prompt, use_audio_in_video=False):
        model = state["model"]
        processor = state["processor"]
        tokenizer = state["tokenizer"]

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_message}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "nframes": 16,
                    },
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        prompt_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs, _ = process_vision_info(
            messages, return_video_kwargs=True
        )

        inputs = processor(
            text=[prompt_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        with torch.inference_mode():
            out_ids = model.generate(
                **inputs,
                temperature=0.7,
                top_p=0.001,
                repetition_penalty=1.05,
                max_new_tokens=2048,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        out_ids_trimmed = [o[len(i) :] for i, o in zip(inputs.input_ids, out_ids)]
        output_text = processor.batch_decode(
            out_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        answer_match = re.search(r"<answer>(.*?)</answer>", output_text, re.DOTALL)
        if answer_match:
            return answer_match.group(1).strip()

        return output_text

    state["fn"] = interrogate
