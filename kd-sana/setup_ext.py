import gc
import random
import shutil
import subprocess
import sys
import tempfile
from uuid import uuid4
import gradio as gr
import os
import re
from pathlib import Path
import numpy as np
import os, torch
import time
from PIL import Image
from torchvision.utils import save_image

title = "Sana"
dir = Path(__file__).parent.absolute()


def setup(kubin):
    sana_model = {"pipe": None}

    def sana_ui(ui_shared, ui_tabs):
        with gr.Row() as sana_block:
            with gr.Row():
                with gr.Column(scale=2) as sana_params_block:
                    with gr.Row():
                        prompt = gr.TextArea(
                            value='A cyberpunk cat with a neon sign that says "Sana"',
                            label="Prompt",
                            lines=2,
                        )
                    with gr.Row():
                        negative_prompt = gr.TextArea(
                            value="",
                            label="Negative prompt",
                            lines=2,
                        )
                    with gr.Row():
                        guidance_scale = gr.Slider(
                            1, 10, 5, step=1, label="Guidance Scale"
                        )
                        pag_guidance_scale = gr.Slider(
                            1, 5, 2, step=1, label="PAG Guidance Scale"
                        )
                        steps = gr.Slider(2, 50, 18, step=1, label="Steps")

                    with gr.Row():
                        width = gr.Slider(512, 4096, 1024, step=64, label="Width")
                        height = gr.Slider(512, 4096, 1024, step=64, label="Height")
                        seed = gr.Number(-1, label="Seed", precision=0)

                    with gr.Row():
                        with gr.Accordion("Utilities", open=False):
                            flush_btn = gr.Button(
                                "Release VRAM", variant="secondary", scale=1
                            )
                            flush_btn.click(
                                fn=lambda: flush(sana_model, kubin),
                                inputs=None,
                                outputs=None,
                            )

                sana_params_block.elem_classes = ["block-params"]

                with gr.Column(scale=1) as sana_output_block:
                    generate_btn = gr.Button("Generate", variant="primary")
                    sana_output = gr.Gallery(
                        label="Generated Images",
                        columns=2,
                        preview=True,
                        elem_classes=["kd-sana-output"],
                    )

                    sana_output.select(
                        fn=None,
                        _js=f"() => kubin.UI.setImageIndex('sana-output')",
                        show_progress=False,
                        outputs=gr.State(None),
                    )

                    ui_shared.create_base_send_targets(
                        sana_output, "sana-output", ui_tabs
                    )
                    ui_shared.create_ext_send_targets(
                        sana_output, "sana-output", ui_tabs
                    )

                kubin.ui_utils.click_and_disable(
                    generate_btn,
                    fn=lambda *params: generate_sana(kubin, sana_model, *params),
                    inputs=[
                        prompt,
                        negative_prompt,
                        steps,
                        guidance_scale,
                        pag_guidance_scale,
                        seed,
                        width,
                        height,
                        gr.Textbox(
                            value=kubin.params("general", "device"), visible=False
                        ),
                        gr.Textbox(
                            value=kubin.params("general", "cache_dir"),
                            visible=False,
                        ),
                        gr.Textbox(
                            value=kubin.params("general", "output_dir"),
                            visible=False,
                        ),
                    ],
                    outputs=sana_output,
                    js=[
                        f"args => kubin.UI.taskStarted('{title}')",
                        f"args => kubin.UI.taskFinished('{title}')",
                    ],
                )

        return sana_block

    return {
        "title": title,
        "tab_ui": lambda ui_s, ts: sana_ui(ui_s, ts),
    }


def peek_hf_key():
    hf_key_path = os.path.join("extensions", "kd-sana", "hf.key")
    if not os.path.exists(hf_key_path):
        print(
            "Some models may be gated. To prevent this, you need to put your HuggingFace API key in extensions/kd-sana/hf.key"
        )
        return

    with open(hf_key_path, "r") as file:
        hf_key = file.read()
    os.environ["HF_TOKEN"] = hf_key


def init_model(kubin, sana_model, device, cache_dir):
    peek_hf_key()

    sys.path.append(os.path.join("extensions", "kd-sana", "sana"))
    from sana.app.sana_pipeline import SanaPipeline

    config_path = Path(
        os.path.join("extensions", "kd-sana", "sana_config.yaml")
    ).resolve()

    pipe = SanaPipeline(config=config_path)
    pipe.from_pretrained(
        "hf://Efficient-Large-Model/Sana_1600M_1024px/checkpoints/Sana_1600M_1024px.pth"
    )
    pipe.register_progress_bar(gr.Progress())

    sana_model["pipe"] = pipe
    return pipe


def generate_sana(
    kubin,
    sana_model,
    prompt,
    negative_prompt,
    steps,
    guidance_scale,
    pag_guidance_scale,
    seed,
    width,
    height,
    device,
    cache_dir,
    output_dir,
):
    sana_pipe = sana_model.get("pipe", None)
    if sana_pipe is None:
        sana_pipe = init_model(kubin, sana_model, device, cache_dir)

    images = sana_pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        pag_guidance_scale=pag_guidance_scale,
        width=width,
        height=height,
        num_images_per_prompt=1,
        generator=torch.Generator(device=device).manual_seed(
            random.randint(0, np.iinfo(np.int32).max) if seed == -1 else seed
        ),
        latents=None,
    )

    imagepaths = []
    for image in images:
        folder_path = os.path.join(output_dir, "sana")
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        path = os.path.join(folder_path, f"sana-{uuid4()}.png")
        save_image(image, path, nrow=1, normalize=True, value_range=(-1, 1))
        imagepaths.append(path)
    return imagepaths


def flush(sana_model, kubin):
    kubin.model.flush(None)
    pipe = sana_model.get("pipe", None)

    if pipe is not None:
        pipe.model.to("cpu")
        sana_model["pipe"] = None

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def mount(kubin):
    sana_repo = "https://github.com/NVlabs/Sana"
    commit = "41dcbe9"
    destination_dir = os.path.join("extensions", "kd-sana", "sana")

    if not os.path.exists(destination_dir):
        print("Cloning Sana github repo...")
        current_path = os.getcwd()
        temp_dir = tempfile.mkdtemp()
        temp_sana = os.path.join(temp_dir, "temp_sana")

        subprocess.run(["git", "clone", sana_repo, temp_sana, "-q"])
        os.chdir(temp_sana)
        subprocess.run(["git", "checkout", commit, "-q"])
        os.chdir(current_path)

        if not os.path.exists(destination_dir):
            shutil.copytree(temp_sana, destination_dir)
            init_file_path = os.path.join(destination_dir, "__init__.py")
            open(init_file_path, "a").close()

        for dirpath, dirnames, filenames in os.walk(destination_dir):
            for filename in filenames:
                if filename.endswith(".py"):
                    file_path = os.path.join(dirpath, filename)

                    with open(file_path, "r", encoding="utf-8") as file:
                        content = file.read()

                    modified_content = re.sub(
                        r"^(from\s+)sana(\.[a-zA-Z_\.]+\s+import\s+)",
                        r"\1sana.sana\2",
                        content,
                        flags=re.MULTILINE,
                    )

                    modified_content = re.sub(
                        r'log_file\s*=\s*"/dev/null"',
                        "log_file = os.devnull",
                        modified_content,
                    )

                    if modified_content != content:
                        with open(file_path, "w", encoding="utf-8") as file:
                            file.write(modified_content)
                        print(f"Patched: {file_path}")

        fcntl_src = os.path.join("extensions", "kd-sana", "fcntl.py")
        fcntl_dest = os.path.join(destination_dir, "fcntl.py")
        if os.path.exists(fcntl_src):
            shutil.copy2(fcntl_src, fcntl_dest)
            print(f"Copied fcntl.py to {fcntl_dest}")

        try:
            shutil.rmtree(temp_dir)
        except:
            pass
