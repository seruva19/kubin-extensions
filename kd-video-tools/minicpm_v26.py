import torch
from transformers import AutoModel, AutoTokenizer
from transformers import BitsAndBytesConfig
from decord import VideoReader, cpu
from PIL import Image

MINICPM_MODEL_ID = "openbmb/MiniCPM-V-2_6"


def init_minicpm(state, device, cache_dir, quantization, use_flash_attention):
    state["name"] = MINICPM_MODEL_ID

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

    state["model"] = AutoModel.from_pretrained(
        MINICPM_MODEL_ID,
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if use_flash_attention else "sdpa",
        torch_dtype=torch.bfloat16,
        quantization_config=q_conf,
        cache_dir=cache_dir,
    )

    if q_conf is None:
        state["model"].to(device=device)
        state["q"] = None
    else:
        state["q"] = q_conf
    state["model"].eval()

    state["tokenizer"] = AutoTokenizer.from_pretrained(
        MINICPM_MODEL_ID, trust_remote_code=True, cache_dir=cache_dir
    )

    MAX_NUM_FRAMES = 512

    def encode_video(video_path):
        def uniform_sample(l, n):
            gap = len(l) / n
            idxs = [int(i * gap + gap / 2) for i in range(n)]
            return [l[i] for i in idxs]

        vr = VideoReader(video_path, ctx=cpu(0))
        sample_fps = round(vr.get_avg_fps() / 1)  # FPS
        frame_idx = [i for i in range(0, len(vr), sample_fps)]
        if len(frame_idx) > MAX_NUM_FRAMES:
            frame_idx = uniform_sample(frame_idx, MAX_NUM_FRAMES)
        frames = vr.get_batch(frame_idx).asnumpy()
        frames = [Image.fromarray(v.astype("uint8")) for v in frames]
        print("num frames:", len(frames))
        return frames

    def interrogate(video_path, question):
        frames = encode_video(video_path)
        msgs = [
            {"role": "user", "content": frames + [question]},
        ]

        params = {}
        params["use_image_id"] = False
        params["max_slice_nums"] = 2

        answer = state["model"].chat(
            image=None, msgs=msgs, tokenizer=state["tokenizer"], **params
        )
        return answer

    state["fn"] = interrogate
