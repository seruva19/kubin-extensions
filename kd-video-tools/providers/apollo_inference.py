import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig

from apollo.mm_utils import KeywordsStoppingCriteria, tokenizer_mm_token, ApolloMMLoader

from apollo.conversation import conv_templates, SeparatorStyle
from huggingface_hub import snapshot_download

APOLLO_MODEL_ID = "GoodiesHere/Apollo-LMMs-Apollo-7B-t32"


def init_apollo(state, device, cache_dir, quantization):
    model_path = snapshot_download(APOLLO_MODEL_ID, repo_type="model")

    state["name"] = APOLLO_MODEL_ID

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

    q_conf = None
    print("warning: apollo does not support quantization yet")

    model = state["model"] = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        quantization_config=q_conf,
        cache_dir=cache_dir,
    )

    tokenizer = state["tokenizer"] = model.tokenizer
    vision_processors = model.vision_tower.vision_processor
    config = model.config
    num_repeat_token = config.mm_connector_cfg["num_output_tokens"]
    mm_processor = ApolloMMLoader(
        vision_processors,
        config.clip_duration,
        frames_per_clip=4,
        clip_sampling_ratio=0.65,
        model_max_length=config.model_max_length,
        device=device,
        num_repeat_token=num_repeat_token,
    )

    if q_conf is None:
        state["model"].to(device=device, dtype=torch.bfloat16)
        state["q"] = None
    else:
        state["q"] = q_conf
    state["model"].eval()

    def interrogate(file_path, question):
        image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}
        file_extension = os.path.splitext(file_path)[1].lower()

        if file_extension in image_extensions:
            mm_data, replace_string = mm_processor.load_image(file_path)
        else:
            mm_data, replace_string = mm_processor.load_video(file_path)

        conv = conv_templates["qwen_2"].copy()
        conv.append_message(conv.roles[0], replace_string + "\n\n" + question)
        conv.append_message(conv.roles[1], None)

        prompt = conv.get_prompt()
        input_ids = (
            tokenizer_mm_token(prompt, state["tokenizer"], return_tensors="pt")
            .unsqueeze(0)
            .to(device)
        )

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                vision_input=[mm_data],
                data_types=["video"],
                do_sample=True,
                temperature=0.4,
                max_new_tokens=256,
                top_p=0.7,
                use_cache=True,
                num_beams=1,
                stopping_criteria=[stopping_criteria],
            )

        pred = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return pred

    state["fn"] = interrogate
