import gc
import os

import torch
import torch
from diffusers import FluxTransformer2DModel, FluxPipeline
from transformers import T5EncoderModel


def peek_hf_key():
    current_file_path = os.path.abspath(__file__)
    current_directory = os.path.dirname(current_file_path)
    hf_key_path = os.path.join(current_directory, "hf.key")
    with open(hf_key_path, "r") as file:
        hf_key = file.read()
    os.environ["HF_TOKEN"] = hf_key


def get_models(
    model_name: str,
    device: torch.device,
    cache_dir: str,
    offload: bool,
    use_hf_key: bool,
):
    from optimum.quanto import freeze, qfloat8, quantize

    if use_hf_key:
        peek_hf_key()

    bfl_repo = "black-forest-labs/FLUX.1-dev"
    dtype = torch.bfloat16

    dit_repo_id = (
        "https://huggingface.co/Kijai/flux-fp8/blob/main/flux1-dev-fp8.safetensors"
        if model_name == "flux-dev"
        else "https://huggingface.co/Kijai/flux-fp8/blob/main/flux1-schnell-fp8.safetensors"
    )
    transformer = FluxTransformer2DModel.from_single_file(
        dit_repo_id, torch_dtype=dtype, cache_dir=cache_dir
    )
    quantize(transformer, weights=qfloat8)
    freeze(transformer)

    text_encoder_2 = T5EncoderModel.from_pretrained(
        bfl_repo, subfolder="text_encoder_2", torch_dtype=dtype, cache_dir=cache_dir
    )
    quantize(text_encoder_2, weights=qfloat8)
    freeze(text_encoder_2)

    pipe = FluxPipeline.from_pretrained(
        bfl_repo,
        transformer=None,
        text_encoder_2=None,
        torch_dtype=dtype,
        cache_dir=cache_dir,
    )
    pipe.transformer = transformer
    pipe.text_encoder_2 = text_encoder_2

    if offload:
        pipe.enable_model_cpu_offload()
    return (pipe, transformer, text_encoder_2)


def text_to_image(
    pipe, prompt, width, height, num_inference_steps, guidance_scale, seed
):
    image = pipe(
        prompt=prompt,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
        output_type="pil",
        num_inference_steps=num_inference_steps,
        generator=torch.Generator("cpu").manual_seed(seed),
    ).images[0]

    return [image]


def flush(flux_model, kubin):
    kubin.model.flush(None)
    pipe, transformer, encoder = flux_model.get("models", (None, None, None))

    if pipe is not None:
        pipe.to("cpu")

    if transformer is not None:
        transformer.to("cpu")

    if encoder is not None:
        encoder.to("cpu")

    gc.collect()
    torch.cuda.empty_cache()
    flux_model["models"] = (None, None, None)
