import gc
import shutil
import subprocess
import tempfile
from typing import Optional
from uuid import uuid4
import gradio as gr
import os
import imageio
from pathlib import Path
from transformers import T5Model, T5Tokenizer, CLIPImageProcessor
from diffusers import StableCascadePriorPipeline, StableCascadeDecoderPipeline
import torch

title = "Stable Cascade"
dir = Path(__file__).parent.absolute()


def setup(kubin):
    sc_model = {"prior": None, "decoder": None}

    def stablecascade_ui(ui_shared, ui_tabs):
        with gr.Row() as sc_block:
            with gr.Column(scale=2) as sc_params_block:
                with gr.Row():
                    prompt = gr.TextArea(
                        value="Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
                        label="Prompt",
                        lines=3,
                    )
                with gr.Row():
                    negative_prompt = gr.TextArea(
                        value="",
                        label="Negative prompt",
                        lines=3,
                    )
                with gr.Row():
                    prior_steps = gr.Slider(
                        1,
                        200,
                        20,
                        step=1,
                        label="Prior steps",
                    )
                    prior_guidance_scale = gr.Slider(
                        1, 30, 4.0, step=1, label="Prior guidance scale"
                    )
                    decoder_steps = gr.Slider(
                        1,
                        200,
                        10,
                        step=1,
                        label="Decoder steps",
                    )
                    decoder_guidance_scale = gr.Slider(
                        0, 30, 0.0, step=1, label="Decoder guidance scale"
                    )
                with gr.Row():
                    width = gr.Slider(
                        256,
                        4096,
                        1024,
                        step=64,
                        label="Width",
                    )
                    height = gr.Slider(
                        256,
                        4096,
                        1024,
                        step=64,
                        label="Height",
                    )

                with gr.Row():
                    with gr.Accordion("Utilities", open=False):
                        flush_btn = gr.Button(
                            "Release VRAM", variant="secondary", scale=1
                        )
                        flush_btn.click(
                            fn=lambda: flush(sc_model, kubin),
                            inputs=None,
                            outputs=None,
                        )

            sc_params_block.elem_classes = ["block-params"]

            with gr.Column(scale=1) as sc_output_block:
                generate_btn = gr.Button("Generate", variant="primary")
                sc_output = gr.Gallery(
                    label="Generated Images",
                    columns=2,
                    preview=True,
                    elem_classes=["kd-sc-output"],
                )

                sc_output.select(
                    fn=None,
                    _js=f"() => kubin.UI.setImageIndex('sc-output')",
                    show_progress=False,
                    outputs=gr.State(None),
                )

                ui_shared.create_base_send_targets(sc_output, "sc-output", ui_tabs)
                ui_shared.create_ext_send_targets(sc_output, "sc-output", ui_tabs)

            kubin.ui_utils.click_and_disable(
                generate_btn,
                fn=lambda *params: generate_sc(kubin, sc_model, *params),
                inputs=[
                    prompt,
                    negative_prompt,
                    width,
                    height,
                    prior_steps,
                    prior_guidance_scale,
                    decoder_steps,
                    decoder_guidance_scale,
                    gr.Textbox(value=kubin.params("general", "device"), visible=False),
                    gr.Textbox(
                        value=kubin.params("general", "cache_dir"), visible=False
                    ),
                    gr.Textbox(
                        value=kubin.params("general", "output_dir"), visible=False
                    ),
                ],
                outputs=sc_output,
                js=[
                    f"args => kubin.UI.taskStarted('{title}')",
                    f"args => kubin.UI.taskFinished('{title}')",
                ],
            )

        return sc_block

    return {
        "title": title,
        "tab_ui": lambda ui_s, ts: stablecascade_ui(ui_s, ts),
    }


def init_model(kubin, sc_model, device, cache_dir):
    prior = StableCascadePriorPipeline.from_pretrained(
        "stabilityai/stable-cascade-prior",
        cache_dir=cache_dir,
        variant="bf16",
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
    )

    decoder = StableCascadeDecoderPipeline.from_pretrained(
        "stabilityai/stable-cascade",
        cache_dir=cache_dir,
        variant="bf16",
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
    )

    sc_model["prior"] = prior
    sc_model["decoder"] = decoder

    return sc_model


def generate_sc(
    kubin,
    sc_model,
    prompt,
    negative_prompt,
    width,
    height,
    prior_steps,
    prior_guidance_scale,
    decoder_steps,
    decoder_guidance_scale,
    device,
    cache_dir,
    output_dir,
):
    model = sc_model
    prior = model.get("prior", None)
    decoder = model.get("decoder", None)

    if prior is None or decoder is None:
        model = init_model(kubin, sc_model, device, cache_dir)

    prior = model["prior"]
    prior.enable_model_cpu_offload()

    prior_output = prior(
        prompt=prompt,
        height=height,
        width=width,
        negative_prompt=negative_prompt,
        guidance_scale=prior_guidance_scale,
        num_images_per_prompt=1,
        num_inference_steps=prior_steps,
    )

    decoder = model["decoder"]
    decoder.enable_model_cpu_offload()

    decoder_output = decoder(
        image_embeddings=prior_output.image_embeddings,
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=decoder_guidance_scale,
        output_type="pil",
        num_inference_steps=decoder_steps,
    ).images

    imagepaths = []
    for image in decoder_output:
        folder_path = os.path.join(output_dir, "stable_cascade")
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        path = os.path.join(folder_path, f"sc-{uuid4()}.png")
        image.save(path)
        imagepaths.append(path)

    return imagepaths


def flush(sc_model, kubin):
    kubin.model.flush(None)

    prior = sc_model.get("prior", None)
    decoder = sc_model.get("decoder", None)

    if decoder is not None:
        sc_model["decoder"].to("cpu")
        sc_model["decoder"] = None

    if prior is not None:
        sc_model["prior"].to("cpu")
        sc_model["prior"] = None

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
