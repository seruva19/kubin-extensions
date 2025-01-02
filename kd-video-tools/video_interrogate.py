import io
import os
import numpy as np
from torch import dtype
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
from decord import cpu, VideoReader, bridge
import transformers
from transformers.generation import GenerationMixin
import types
from apollo_inference import APOLLO_MODEL_ID, init_apollo


def init_interrogate_fn(kubin, state, cache_dir, device, model_name, quantization):
    current_model = state["model"]
    current_model_name = state["name"]

    if current_model is None or model_name != current_model_name:
        flush(kubin, state)

        if model_name == "THUDM/cogvlm2-video-llama3-chat":
            state["name"] = "THUDM/cogvlm2-video-llama3-chat"
            state["tokenizer"] = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path="THUDM/cogvlm2-video-llama3-chat",
                trust_remote_code=True,
                cache_dir=cache_dir,
            )

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

            state["model"].eval()
            if q_conf is None:
                state["model"].to(device)
                state["q"] = None
            else:
                state["q"] = q_conf

            def interrogate(video_path, prompt):
                if not video_path.endswith(".mp4"):
                    return "Only mp4 videos are supported."

                strategy = "chat"
                video = load_video(video_path, strategy=strategy)

                history = []
                query = prompt
                temperature = 0.2
                inputs = state["model"].build_conversation_input_ids(
                    tokenizer=state["tokenizer"],
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
                    outputs = state["model"].generate(**inputs, **gen_kwargs)
                    outputs = outputs[:, inputs["input_ids"].shape[1] :]
                    response = state["tokenizer"].decode(
                        outputs[0], skip_special_tokens=True
                    )
                    return response

            state["fn"] = interrogate

        elif model_name == APOLLO_MODEL_ID:
            init_apollo(state, device, cache_dir, quantization)
        else:
            raise ValueError(f"unknown model name: {model_name}")


def flush(kubin, state):
    if state["model"] is not None:
        if state["q"] is None:
            state["model"].to("cpu")
            state["model"] = None
            state["tokenizer"] = None

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