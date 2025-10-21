import torch
from transformers import AutoModel, AutoTokenizer
from transformers import BitsAndBytesConfig


VIDEOCHAT_MODEL_ID = "OpenGVLab/VideoChat-Flash-Qwen2_5-2B_res448"


def init_videochat(state, device, cache_dir, quantization):
    state["name"] = VIDEOCHAT_MODEL_ID

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

    state["tokenizer"] = AutoTokenizer.from_pretrained(
        VIDEOCHAT_MODEL_ID, trust_remote_code=True, cache_dir=cache_dir
    )
    state["model"] = AutoModel.from_pretrained(
        VIDEOCHAT_MODEL_ID,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        quantization_config=q_conf,
        cache_dir=cache_dir,
    )

    tokenizer = state["tokenizer"]
    model = state["model"]

    if q_conf is None:
        state["model"].half().to(device=device)
        state["q"] = None
    else:
        state["q"] = q_conf

    state["image_processor"] = model.get_vision_tower().image_processor
    mm_llm_compress = False
    if mm_llm_compress:
        model.config.mm_llm_compress = True
        model.config.llm_compress_type = "uniform0_attention"
        model.config.llm_compress_layer_list = [4, 18]
        model.config.llm_image_token_ratio_list = [1, 0.75, 0.25]
    else:
        model.config.mm_llm_compress = False

    MAX_NUM_FRAMES = 512
    generation_config = dict(
        do_sample=False, temperature=0.0, max_new_tokens=1024, top_p=0.1, num_beams=1
    )

    def interrogate(video_path, question):
        output, chat_history = model.chat(
            video_path=video_path,
            tokenizer=tokenizer,
            user_prompt=question,
            return_history=True,
            max_num_frames=MAX_NUM_FRAMES,
            generation_config=generation_config,
        )

        return output

    state["fn"] = interrogate
