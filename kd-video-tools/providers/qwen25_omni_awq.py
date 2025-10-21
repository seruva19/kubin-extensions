import os
import torch
import sys
import importlib.util
import soundfile as sf

try:
    from awq.models.base import BaseAWQForCausalLM

    AWQ_AVAILABLE = True
except Exception as e:
    print(
        "Cannot import awq.\nPlease install awq package: pip install git+https://github.com/tiger-of-shawn/AutoAWQ_V2.git --no-deps"
    )
    AWQ_AVAILABLE = False
    BaseAWQForCausalLM = object

from transformers import Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
from huggingface_hub import hf_hub_download

QWEN25_OMNI_AWQ_MODEL_ID = "Qwen/Qwen2.5-Omni-7B-AWQ"


def replace_transformers_module():
    original_mod_name = "transformers.models.qwen2_5_omni.modeling_qwen2_5_omni"
    new_mod_path = "modeling_qwen2_5_omni_low_VRAM_mode.py"

    if original_mod_name in sys.modules:
        del sys.modules[original_mod_name]

    try:
        spec = importlib.util.spec_from_file_location(original_mod_name, new_mod_path)
        new_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(new_mod)
        sys.modules[original_mod_name] = new_mod
    except Exception as e:
        print(f"Warning: Could not load low VRAM mode module: {e}")


class Qwen2_5_OmniAWQForConditionalGeneration(BaseAWQForCausalLM):
    layer_type = "Qwen2_5OmniDecoderLayer"
    max_seq_len_key = "max_position_embeddings"
    modules_to_not_convert = ["visual"]

    @staticmethod
    def get_model_layers(model):
        return model.thinker.model.layers

    @staticmethod
    def get_act_for_scaling(module):
        return dict(is_scalable=False)

    @staticmethod
    def move_embed(model, device: str):
        model.thinker.model.embed_tokens = model.thinker.model.embed_tokens.to(device)
        model.thinker.visual = model.thinker.visual.to(device)
        model.thinker.audio_tower = model.thinker.audio_tower.to(device)

        model.thinker.visual.rotary_pos_emb = model.thinker.visual.rotary_pos_emb.to(
            device
        )
        model.thinker.model.rotary_emb = model.thinker.model.rotary_emb.to(device)

        for layer in model.thinker.model.layers:
            layer.self_attn.rotary_emb = layer.self_attn.rotary_emb.to(device)

    @staticmethod
    def get_layers_for_scaling(module, input_feat, module_kwargs):
        layers = []

        layers.append(
            dict(
                prev_op=module.input_layernorm,
                layers=[
                    module.self_attn.q_proj,
                    module.self_attn.k_proj,
                    module.self_attn.v_proj,
                ],
                inp=input_feat["self_attn.q_proj"],
                module2inspect=module.self_attn,
                kwargs=module_kwargs,
            )
        )

        if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
            layers.append(
                dict(
                    prev_op=module.self_attn.v_proj,
                    layers=[module.self_attn.o_proj],
                    inp=input_feat["self_attn.o_proj"],
                )
            )

        layers.append(
            dict(
                prev_op=module.post_attention_layernorm,
                layers=[module.mlp.gate_proj, module.mlp.up_proj],
                inp=input_feat["mlp.gate_proj"],
                module2inspect=module.mlp,
            )
        )

        layers.append(
            dict(
                prev_op=module.mlp.up_proj,
                layers=[module.mlp.down_proj],
                inp=input_feat["mlp.down_proj"],
            )
        )

        return layers


def init_qwen25_omni_awq(state, device, cache_dir, quantization, use_flash_attention):
    if not AWQ_AVAILABLE:
        raise ImportError(
            "AWQ is not available. Please install: pip install git+https://github.com/tiger-of-shawn/AutoAWQ_V2.git --no-deps"
        )

    replace_transformers_module()

    state["name"] = QWEN25_OMNI_AWQ_MODEL_ID

    try:
        model = Qwen2_5_OmniAWQForConditionalGeneration.from_quantized(
            QWEN25_OMNI_AWQ_MODEL_ID,
            model_type="qwen2_5_omni",
            torch_dtype=torch.float16,
            cache_dir=cache_dir,
            attn_implementation="flash_attention_2" if use_flash_attention else "sdpa",
        )

        try:
            spk_path = hf_hub_download(
                cache_dir=cache_dir,
                repo_id=QWEN25_OMNI_AWQ_MODEL_ID,
                filename="spk_dict.pt",
            )
            model.model.load_speakers(spk_path)
        except Exception as e:
            print(f"Warning: Could not load speaker dictionary: {e}")

        model.model.thinker.model.embed_tokens = (
            model.model.thinker.model.embed_tokens.to(device)
        )
        model.model.thinker.visual = model.model.thinker.visual.to(device)
        model.model.thinker.audio_tower = model.model.thinker.audio_tower.to(device)
        model.model.thinker.visual.rotary_pos_emb = (
            model.model.thinker.visual.rotary_pos_emb.to(device)
        )
        model.model.thinker.model.rotary_emb = model.model.thinker.model.rotary_emb.to(
            device
        )

        for layer in model.model.thinker.model.layers:
            layer.self_attn.rotary_emb = layer.self_attn.rotary_emb.to(device)

        state["model"] = model

        processor = Qwen2_5OmniProcessor.from_pretrained(
            QWEN25_OMNI_AWQ_MODEL_ID, cache_dir=cache_dir
        )
        state["processor"] = processor

        def interrogate(file_path, question, use_audio_in_video=False):

            model = state["model"]
            processor = state["processor"]

            if not question:
                question = "Describe this video in detail."

            system_prompt = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech. Please respond in English."

            image_extensions = {
                ".jpg",
                ".jpeg",
                ".png",
                ".gif",
                ".bmp",
                ".tiff",
                ".webp",
            }
            file_extension = os.path.splitext(file_path)[1].lower()

            if file_extension in image_extensions:
                messages = [
                    {
                        "role": "system",
                        "content": [
                            {"type": "text", "text": system_prompt},
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": file_path},
                            {"type": "text", "text": question},
                        ],
                    },
                ]
            else:
                messages = [
                    {
                        "role": "system",
                        "content": [
                            {"type": "text", "text": system_prompt},
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "video": file_path},
                            {"type": "text", "text": question},
                        ],
                    },
                ]

            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            audios, images, videos = process_mm_info(
                messages, use_audio_in_video=use_audio_in_video
            )
            inputs = processor(
                text=text,
                audio=audios,
                images=images,
                videos=videos,
                return_tensors="pt",
                padding=True,
            )
            inputs = inputs.to(device)

            input_length = inputs["input_ids"].shape[1]

            with torch.inference_mode():
                output = model.generate(
                    **inputs,
                    use_audio_in_video=use_audio_in_video,
                    return_audio=False,
                    max_new_tokens=512,
                )

            response_tokens = output[0][input_length:]
            text_output = processor.batch_decode(
                [response_tokens],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            return text_output

        state["fn"] = interrogate

    except Exception as e:
        print(f"Error initializing Qwen2.5-Omni AWQ model: {e}")
        raise e
