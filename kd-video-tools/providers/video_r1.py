import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor,
)
from transformers import BitsAndBytesConfig
import cv2
from qwen_vl_utils import process_vision_info

VIDEOR1_MODEL_ID = "Video-R1/Video-R1-7B"


def init_videor1(state, device, cache_dir, quantization, use_flash_attention):
    state["name"] = VIDEOR1_MODEL_ID

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

    state["model"] = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        VIDEOR1_MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if use_flash_attention else "sdpa",
        quantization_config=q_conf,
        cache_dir=cache_dir,
    )

    state["processor"] = AutoProcessor.from_pretrained(
        VIDEOR1_MODEL_ID, trust_remote_code=True, cache_dir=cache_dir
    )

    processor = state["processor"]
    processor.image_processor.image_processor_type = "Qwen2VLImageProcessor"

    state["tokenizer"] = AutoTokenizer.from_pretrained(
        VIDEOR1_MODEL_ID, trust_remote_code=True, cache_dir=cache_dir
    )

    tokenizer = state["tokenizer"]
    tokenizer.padding_side = "left"

    processor.tokenizer = tokenizer

    model = state["model"]
    if q_conf is None:
        state["model"].to(device=device)
        state["q"] = None
    else:
        state["q"] = q_conf

    def interrogate(video_path, question):
        QUESTION_TEMPLATE = (
            "{Question}\n"
            "Please think about this question as if you were a human pondering deeply. "
            "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions. "
            "It's encouraged to include self-reflection or verification in the reasoning process. "
            "Provide your detailed reasoning between the <think> and </think> tags, and then give your final answer between the <answer> and </answer> tags."
        )
        TYPE_TEMPLATE = {
            "free-form": " Please provide your text answer within the <answer> </answer> tags."
        }

        problem_type = "free-form"
        IMAGE_FACTOR = 28
        MIN_PIXELS = 4 * 28 * 28
        MAX_PIXELS = 256 * 28 * 28
        MAX_RATIO = 200

        VIDEO_MIN_PIXELS = 128 * 28 * 28
        VIDEO_MAX_PIXELS = 128 * 28 * 28
        FRAME_FACTOR = 2
        FPS = 2.0
        FPS_MIN_FRAMES = 4
        FPS_MAX_FRAMES = 16
        nframes = 32

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "max_pixels": VIDEO_MAX_PIXELS,
                        "fps": FPS,
                        "nframes": nframes,
                    },
                    {
                        "type": "text",
                        "text": QUESTION_TEMPLATE.format(Question=question)
                        + TYPE_TEMPLATE[problem_type],
                    },
                ],
            }
        ]
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(device)

        generated_ids = model.generate(
            **inputs, do_sample=True, temperature=0.1, top_p=0.001, max_new_tokens=256
        )

        input_length = inputs["input_ids"].shape[1]
        generated_ids_trimmed = generated_ids[:, input_length:]

        output_texts = tokenizer.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        print(output_texts[0])
        return output_texts[0]

    state["fn"] = interrogate
