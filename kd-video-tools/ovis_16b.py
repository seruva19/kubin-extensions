import torch
from PIL import Image
from transformers import AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from moviepy.editor import VideoFileClip

OVIS2_MODEL_ID = "AIDC-AI/Ovis2-16B"


def init_ovis2(state, device, cache_dir, quantization, use_flash_attention):
    state["name"] = OVIS2_MODEL_ID

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
        OVIS2_MODEL_ID,
        torch_dtype=torch.bfloat16,
        multimodal_max_length=32768,
        trust_remote_code=True,
        # attn_implementation="flash_attention_2" if use_flash_attention else "sdpa",
        quantization_config=q_conf,
        cache_dir=cache_dir,
    )

    if q_conf is None:
        state["model"].to(device).eval()

    state["text_tokenizer"] = state["model"].get_text_tokenizer()
    state["visual_tokenizer"] = state["model"].get_visual_tokenizer()

    def interrogate(video_path, question):
        num_frames = 12
        max_partition = 1

        with VideoFileClip(video_path) as clip:
            total_frames = int(clip.fps * clip.duration)
            if total_frames <= num_frames:
                sampled_indices = range(total_frames)
            else:
                stride = total_frames / num_frames
                sampled_indices = [
                    min(total_frames - 1, int((stride * i + stride * (i + 1)) / 2))
                    for i in range(num_frames)
                ]
            frames = [clip.get_frame(index / clip.fps) for index in sampled_indices]
            frames = [Image.fromarray(frame, mode="RGB") for frame in frames]
        images = frames
        query = "\n".join(["<image>"] * len(images)) + "\n" + question

        model = state["model"]
        text_tokenizer = state["text_tokenizer"]
        visual_tokenizer = state["visual_tokenizer"]

        prompt, input_ids, pixel_values = model.preprocess_inputs(
            query, images, max_partition=max_partition
        )
        attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
        input_ids = input_ids.unsqueeze(0).to(device=model.device)
        attention_mask = attention_mask.unsqueeze(0).to(device=model.device)
        if pixel_values is not None:
            pixel_values = pixel_values.to(
                dtype=visual_tokenizer.dtype, device=visual_tokenizer.device
            )
        pixel_values = [pixel_values]

        with torch.inference_mode():
            gen_kwargs = dict(
                max_new_tokens=1024,
                do_sample=False,
                top_p=None,
                top_k=None,
                temperature=None,
                repetition_penalty=None,
                eos_token_id=model.generation_config.eos_token_id,
                pad_token_id=text_tokenizer.pad_token_id,
                use_cache=True,
            )
            output_ids = model.generate(
                input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                **gen_kwargs,
            )[0]
            output = text_tokenizer.decode(output_ids, skip_special_tokens=True)
            return output

    state["fn"] = interrogate
