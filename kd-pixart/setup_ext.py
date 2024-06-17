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
from diffusers import Transformer2DModel, PixArtSigmaPipeline
from torch import nn
import torch

title = "PixArt Sigma"
dir = Path(__file__).parent.absolute()


def setup(kubin):
    pixart_model = {"transformer": None, "pipe": None}

    def pixart_ui(ui_shared, ui_tabs):
        with gr.Row() as pixart_block:
            with gr.Column(scale=2) as pixart_params_block:
                with gr.Row():
                    prompt = gr.TextArea(
                        value="Body shot, a French woman, Photography, French Streets background, backlight, rim light, Fujifilm",
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
                    steps = gr.Slider(
                        1,
                        200,
                        20,
                        step=1,
                        label="Steps",
                    )
                    guidance_scale = gr.Slider(
                        1, 30, 4.5, step=1, label="Guidance scale"
                    )
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
                            fn=lambda: flush(pixart_model, kubin),
                            inputs=None,
                            outputs=None,
                        )

            pixart_params_block.elem_classes = ["block-params"]

            with gr.Column(scale=1) as pixart_output_block:
                generate_btn = gr.Button("Generate", variant="primary")
                pixart_output = gr.Gallery(
                    label="Generated Images",
                    columns=2,
                    preview=True,
                    elem_classes=["kd-pixart-output"],
                )

                pixart_output.select(
                    fn=None,
                    _js=f"() => kubin.UI.setImageIndex('pixart-output')",
                    show_progress=False,
                    outputs=gr.State(None),
                )

                ui_shared.create_base_send_targets(
                    pixart_output, "pixart-output", ui_tabs
                )
                ui_shared.create_ext_send_targets(
                    pixart_output, "pixart-output", ui_tabs
                )

            kubin.ui_utils.click_and_disable(
                generate_btn,
                fn=lambda *params: generate_pixart(kubin, pixart_model, *params),
                inputs=[
                    prompt,
                    negative_prompt,
                    width,
                    height,
                    steps,
                    guidance_scale,
                    gr.Textbox(value=kubin.params("general", "device"), visible=False),
                    gr.Textbox(
                        value=kubin.params("general", "cache_dir"), visible=False
                    ),
                    gr.Textbox(
                        value=kubin.params("general", "output_dir"), visible=False
                    ),
                ],
                outputs=pixart_output,
                js=[
                    f"args => kubin.UI.taskStarted('{title}')",
                    f"args => kubin.UI.taskFinished('{title}')",
                ],
            )

        return pixart_block

    return {
        "title": title,
        "tab_ui": lambda ui_s, ts: pixart_ui(ui_s, ts),
    }


def init_model(kubin, pixart_model, device, cache_dir):
    pixart_transformer = Transformer2DModel.from_pretrained(
        "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
        cache_dir=cache_dir,
        subfolder="transformer",
        torch_dtype=torch.float16,
        use_safetensors=True,
    )

    pixart_pipe = PixArtSigmaPipeline.from_pretrained(
        "PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers",
        cache_dir=cache_dir,
        transformer=pixart_transformer,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )

    pixart_pipe.to(device)

    pixart_model["transformer"] = pixart_transformer
    pixart_model["pipe"] = pixart_pipe

    return pixart_pipe


def generate_pixart(
    kubin,
    pixart_model,
    prompt,
    negative_prompt,
    width,
    height,
    steps,
    guidance_scale,
    device,
    cache_dir,
    output_dir,
):
    pixart_pipe = pixart_model.get("pipe", None)
    if pixart_pipe is None:
        pixart_pipe = init_model(kubin, pixart_model, device, cache_dir)

    images = pixart_pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
        clean_caption=False,
        num_images_per_prompt=1,
    ).images

    imagepaths = []
    for image in images:
        path = os.path.join(output_dir, f"pixart-{uuid4()}.png")
        image.save(path)
        imagepaths.append(path)

    return imagepaths


def flush(pixart_model, kubin):
    kubin.model.flush(None)
    pipe = pixart_model.get("pipe", None)

    if pipe is not None:
        pixart_model["transformer"].to("cpu")
        pixart_model["transformer"] = None

        pixart_model["pipe"].to("cpu")
        pixart_model["pipe"] = None

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
