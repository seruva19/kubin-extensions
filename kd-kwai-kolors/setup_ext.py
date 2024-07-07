import gc
import shutil
import subprocess
import tempfile
from uuid import uuid4
import gradio as gr
import os
from pathlib import Path
import os, torch
from diffusers import UNet2DConditionModel, AutoencoderKL
from diffusers import EulerDiscreteScheduler
from huggingface_hub import hf_hub_download, snapshot_download

title = "Kwai Kolors"
dir = Path(__file__).parent.absolute()


def setup(kubin):
    kwai_model = {"pipe": None}

    def kwai_ui(ui_shared, ui_tabs):
        with gr.Row() as kwai_block:
            with gr.Column(scale=2) as kwai_params_block:
                with gr.Row():
                    prompt = gr.TextArea(
                        value="Anime girl, detailed, 4k",
                        label="Prompt",
                        lines=3,
                    )
                with gr.Row():
                    negative_prompt = gr.TextArea(
                        value="", label="Negative prompt", lines=3, interactive=False
                    )
                with gr.Row():
                    steps = gr.Slider(
                        1,
                        200,
                        50,
                        step=1,
                        label="Steps",
                    )
                    guidance_scale = gr.Slider(
                        1, 30, 5.0, step=1, label="Guidance scale"
                    )
                    seed = gr.Number(-1, label="Seed", precision=0)
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
                            fn=lambda: flush(kwai_model, kubin),
                            inputs=None,
                            outputs=None,
                        )

            kwai_params_block.elem_classes = ["block-params"]

            with gr.Column(scale=1) as kwai_output_block:
                generate_btn = gr.Button("Generate", variant="primary")
                kwai_output = gr.Gallery(
                    label="Generated Images",
                    columns=2,
                    preview=True,
                    elem_classes=["kd-kwai-output"],
                )

                kwai_output.select(
                    fn=None,
                    _js=f"() => kubin.UI.setImageIndex('kwai-output')",
                    show_progress=False,
                    outputs=gr.State(None),
                )

                ui_shared.create_base_send_targets(kwai_output, "kwai-output", ui_tabs)
                ui_shared.create_ext_send_targets(kwai_output, "kwai-output", ui_tabs)

            kubin.ui_utils.click_and_disable(
                generate_btn,
                fn=lambda *params: generate_kwai(kubin, kwai_model, *params),
                inputs=[
                    prompt,
                    negative_prompt,
                    width,
                    height,
                    steps,
                    guidance_scale,
                    seed,
                    gr.Textbox(value=kubin.params("general", "device"), visible=False),
                    gr.Textbox(
                        value=kubin.params("general", "cache_dir"), visible=False
                    ),
                    gr.Textbox(
                        value=kubin.params("general", "output_dir"), visible=False
                    ),
                ],
                outputs=kwai_output,
                js=[
                    f"args => kubin.UI.taskStarted('{title}')",
                    f"args => kubin.UI.taskFinished('{title}')",
                ],
            )

        return kwai_block

    return {
        "title": title,
        "tab_ui": lambda ui_s, ts: kwai_ui(ui_s, ts),
    }


def init_model(kubin, kwai_model, device, cache_dir):
    from kolors.pipelines.pipeline_stable_diffusion_xl_chatglm_256 import (
        StableDiffusionXLPipeline,
    )
    from kolors.models.modeling_chatglm import ChatGLMModel
    from kolors.models.tokenization_chatglm import ChatGLMTokenizer

    ckpt_dir = f"{cache_dir}/Kolors"

    snapshot_download(
        resume_download=True,
        repo_id="Kwai-Kolors/Kolors",
        local_dir=ckpt_dir,
        ignore_patterns=["diffusion_pytorch_model.*"],
        allow_patterns=["diffusion_pytorch_model.fp16.safetensors"],
        local_dir_use_symlinks=False,
    )

    text_encoder = ChatGLMModel.from_pretrained(
        f"{ckpt_dir}/text_encoder", torch_dtype=torch.float16
    ).half()

    tokenizer = ChatGLMTokenizer.from_pretrained(f"{ckpt_dir}/text_encoder")

    vae = AutoencoderKL.from_pretrained(
        f"{ckpt_dir}/vae", revision=None, use_safetensors=True, variant="fp16"
    ).half()

    scheduler = EulerDiscreteScheduler.from_pretrained(f"{ckpt_dir}/scheduler")

    unet = UNet2DConditionModel.from_pretrained(
        f"{ckpt_dir}/unet", revision=None, use_safetensors=True, variant="fp16"
    ).half()

    pipe = StableDiffusionXLPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        force_zeros_for_empty_prompt=False,
    )

    pipe = pipe.to(device)
    pipe.enable_model_cpu_offload()

    kwai_model["pipe"] = pipe
    return pipe


def generate_kwai(
    kubin,
    kwai_model,
    prompt,
    negative_prompt,
    width,
    height,
    steps,
    guidance_scale,
    seed,
    device,
    cache_dir,
    output_dir,
):
    kwai_pipe = kwai_model.get("pipe", None)
    if kwai_pipe is None:
        kwai_pipe = init_model(kubin, kwai_model, device, cache_dir)

    images = kwai_pipe(
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        num_images_per_prompt=1,
        generator=torch.Generator(kwai_pipe.device).manual_seed(seed),
    ).images

    imagepaths = []
    for image in images:
        folder_path = os.path.join(output_dir, "kwai_kolors")
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        path = os.path.join(folder_path, f"kolors-{uuid4()}.png")
        image.save(path)
        imagepaths.append(path)

    return imagepaths


def flush(kwai_model, kubin):
    kubin.model.flush(None)
    pipe = kwai_model.get("pipe", None)

    if pipe is not None:
        kwai_model["pipe"].to("cpu")
        kwai_model["pipe"] = None

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def mount(kubin):
    kwai_repo = "https://github.com/Kwai-Kolors/Kolors"
    commit = "f9c664c"
    destination_dir = "extensions/kd-kwai-kolors/kolors"

    if not os.path.exists(destination_dir):
        current_path = os.getcwd()
        temp_dir = tempfile.mkdtemp()
        temp_kdkwai = os.path.join(temp_dir, "temp_kwai")

        subprocess.run(["git", "clone", kwai_repo, temp_kdkwai, "-q"])
        os.chdir(temp_kdkwai)
        subprocess.run(["git", "checkout", commit, "-q"])
        os.chdir(current_path)

        repo_path = os.path.join(temp_kdkwai, "kolors")
        if not os.path.exists(destination_dir):
            shutil.copytree(repo_path, destination_dir)
        try:
            shutil.rmtree(temp_dir)
        except:
            pass
