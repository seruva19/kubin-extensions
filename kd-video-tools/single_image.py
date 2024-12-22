import io
import numpy as np
from torch import dtype
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
from decord import cpu, VideoReader, bridge
import transformers
from transformers.generation import GenerationMixin
import types


def interrogate_single_image(
    kubin,
    state,
    cache_dir,
    device,
    model_name,
    quantization,
    input_video,
    input_prompt,
    prepended="",
    appended="",
):
    model_id = state["name"]
    model = state["model"]

    if model is None or model_name != model_id:
        flush(kubin, state)

        if model_name == "THUDM/cogvlm2-video-llama3-chat":
            state["name"] = "THUDM/cogvlm2-video-llama3-chat"
            state["tokenizer"] = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path="THUDM/cogvlm2-video-llama3-chat",
                trust_remote_code=True,
                cache_dir=cache_dir,
            )
            tokenizer = state["tokenizer"]

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
                pretrained_model_name_or_path="THUDM/cogvlm2-video-llama3-chat",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                cache_dir=cache_dir,
                quantization_config=q_conf,
            )
            model_name = state["model"]
            model_name.eval()
            if q_conf is None:
                model_name.to(device)

            def interrogate(in_video, prompt):
                strategy = "chat"
                video = load_video(in_video, strategy=strategy)

                history = []
                query = prompt
                temperature = 0.2
                inputs = model_name.build_conversation_input_ids(
                    tokenizer=tokenizer,
                    query=query,
                    images=[video],
                    history=history,
                    template_version=strategy,
                )

                inputs = {
                    "input_ids": inputs["input_ids"].unsqueeze(0).to(device),
                    "token_type_ids": inputs["token_type_ids"].unsqueeze(0).to(device),
                    "attention_mask": inputs["attention_mask"].unsqueeze(0).to(device),
                    "images": [
                        [
                            inputs["images"][0]
                            .to(device)
                            .to(
                                torch.bfloat16
                                if torch.cuda.is_available()
                                and torch.cuda.get_device_capability()[0] >= 8
                                else torch.float16
                            )
                        ]
                    ],
                }

                gen_kwargs = {
                    "max_new_tokens": 2048,
                    "pad_token_id": 128002,
                    "top_k": 1,
                    "do_sample": False,
                    "top_p": 0.1,
                    "temperature": temperature,
                }

                with torch.no_grad():
                    outputs = model_name.generate(**inputs, **gen_kwargs)
                    outputs = outputs[:, inputs["input_ids"].shape[1] :]
                    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    return response

            state["fn"] = interrogate
        else:
            pass

    return prepended + state["fn"](input_video, input_prompt) + appended


def flush(kubin, now):
    if now["model"] is not None:
        now["model"].to("cpu")
        del now["tokenizer"]

    kubin.model.flush(None)


def load_video(video_path, strategy="chat"):
    bridge.set_bridge("torch")
    num_frames = 24

    with open(video_path, "rb") as f:
        mp4_stream = f.read()
    video_stream = io.BytesIO(mp4_stream)

    decord_vr = VideoReader(video_stream, ctx=cpu(0))

    frame_id_list = None
    total_frames = len(decord_vr)
    if strategy == "base":
        clip_end_sec = 60
        clip_start_sec = 0
        start_frame = int(clip_start_sec * decord_vr.get_avg_fps())
        end_frame = (
            min(total_frames, int(clip_end_sec * decord_vr.get_avg_fps()))
            if clip_end_sec is not None
            else total_frames
        )
        frame_id_list = np.linspace(start_frame, end_frame - 1, num_frames, dtype=int)

    elif strategy == "chat":
        timestamps = decord_vr.get_frame_timestamp(np.arange(total_frames))
        timestamps = [i[0] for i in timestamps]
        max_second = round(max(timestamps)) + 1
        frame_id_list = []
        for second in range(max_second):
            closest_num = min(timestamps, key=lambda x: abs(x - second))
            index = timestamps.index(closest_num)
            frame_id_list.append(index)
            if len(frame_id_list) >= num_frames:
                break
    video_data = decord_vr.get_batch(frame_id_list)
    video_data = video_data.permute(3, 0, 1, 2)
    return video_data
