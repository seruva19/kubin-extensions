import gc
import shutil
import subprocess
import sys
import tempfile
from uuid import uuid4
import gradio as gr
import os
import re
from pathlib import Path
import os, torch


title = "Switti"
dir = Path(__file__).parent.absolute()


def setup(kubin):
    switti_model = {"pipe": None}

    def switti_ui(ui_shared, ui_tabs):
        with gr.Row() as switti_block:
            with gr.Tab(label="Inference"):
                with gr.Row():
                    with gr.Column(scale=2) as switti_params_block:
                        with gr.Row():
                            prompt = gr.TextArea(
                                value="Cute winter dragon baby, kawaii, Pixar, ultra detailed, glacial background, extremely realistic",
                                label="Text prompt",
                                lines=2,
                            )
                        with gr.Row():
                            negative_prompt = gr.TextArea(
                                value="",
                                label="Negative prompt for CFG",
                                lines=2,
                            )
                        with gr.Row():
                            cls_guidance = gr.Slider(
                                1, 30, 6, step=1, label="CFG ratio"
                            )
                            top_k_sampling = gr.Slider(
                                1, 500, 400, step=1, label="Top-k sampling"
                            )
                            top_p_sampling = gr.Slider(
                                0, 1, 0.95, step=0.05, label="Top-p sampling"
                            )
                        with gr.Row():
                            seed = gr.Number(-1, label="Seed", precision=0)
                            width = gr.Slider(
                                256,
                                4096,
                                512,
                                step=64,
                                label="Width",
                            )
                            height = gr.Slider(
                                256,
                                4096,
                                512,
                                step=64,
                                label="Height",
                            )
                            more_smooth = gr.Checkbox(
                                True,
                                label="Use gumbel softmax for sampling",
                            )
                        with gr.Row():
                            smooth_start_si = gr.Slider(
                                0, 100, 2, step=1, label="Smoothing starting scale"
                            )
                            last_scale_temp = gr.Slider(
                                0,
                                1,
                                0.1,
                                step=0.05,
                                label="Temperature after disabling CFG",
                            )
                        with gr.Row():
                            turn_on_cfg_start_si = gr.Slider(
                                0, 100, 2, step=1, label="Enable CFG starting scale"
                            )
                            turn_off_cfg_start_si = gr.Slider(
                                0, 100, 8, step=1, label="Disable CFG starting scale"
                            )

                        with gr.Row():
                            with gr.Accordion("Utilities", open=False):
                                flush_btn = gr.Button(
                                    "Release VRAM", variant="secondary", scale=1
                                )
                                flush_btn.click(
                                    fn=lambda: flush(switti_model, kubin),
                                    inputs=None,
                                    outputs=None,
                                )

                    switti_params_block.elem_classes = ["block-params"]

                    with gr.Column(scale=1) as switti_output_block:
                        generate_btn = gr.Button("Generate", variant="primary")
                        switti_output = gr.Gallery(
                            label="Generated Images",
                            columns=2,
                            preview=True,
                            elem_classes=["kd-switti-output"],
                        )

                        switti_output.select(
                            fn=None,
                            _js=f"() => kubin.UI.setImageIndex('switti-output')",
                            show_progress=False,
                            outputs=gr.State(None),
                        )

                        ui_shared.create_base_send_targets(
                            switti_output, "switti-output", ui_tabs
                        )
                        ui_shared.create_ext_send_targets(
                            switti_output, "switti-output", ui_tabs
                        )

                    kubin.ui_utils.click_and_disable(
                        generate_btn,
                        fn=lambda *params: generate_switti(
                            kubin, switti_model, *params
                        ),
                        inputs=[
                            prompt,
                            negative_prompt,
                            seed,
                            cls_guidance,
                            top_k_sampling,
                            top_p_sampling,
                            more_smooth,
                            smooth_start_si,
                            turn_on_cfg_start_si,
                            turn_off_cfg_start_si,
                            last_scale_temp,
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
                        outputs=switti_output,
                        js=[
                            f"args => kubin.UI.taskStarted('{title}')",
                            f"args => kubin.UI.taskFinished('{title}')",
                        ],
                    )

        return switti_block

    return {
        "title": title,
        "tab_ui": lambda ui_s, ts: switti_ui(ui_s, ts),
    }


def init_model(kubin, switti_model, device, cache_dir):
    from switti.models import SwittiPipeline

    model_path = "yresearch/Switti"
    pipe = SwittiPipeline.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device=device
    )
    switti_model["pipe"] = pipe
    return pipe


def generate_switti(
    kubin,
    switti_model,
    prompt,
    negative_prompt,
    seed,
    cls_guidance,
    top_k_sampling,
    top_p_sampling,
    more_smooth,
    smooth_start_si,
    turn_on_cfg_start_si,
    turn_off_cfg_start_si,
    last_scale_temp,
    width,
    height,
    device,
    cache_dir,
    output_dir,
):

    switti_pipe = switti_model.get("pipe", None)
    if switti_pipe is None:
        switti_pipe = init_model(kubin, switti_model, device, cache_dir)

    images = switti_pipe(
        prompt=prompt,
        null_prompt=negative_prompt,
        cfg=cls_guidance,
        top_k=top_k_sampling,
        top_p=top_p_sampling,
        more_smooth=more_smooth,
        smooth_start_si=smooth_start_si,
        turn_on_cfg_start_si=turn_on_cfg_start_si,
        turn_off_cfg_start_si=turn_off_cfg_start_si,
        last_scale_temp=last_scale_temp,
        image_size=(width, height),
        seed=seed,
    )

    imagepaths = []
    for image in images:
        folder_path = os.path.join(output_dir, "switti")
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        path = os.path.join(folder_path, f"switti-{uuid4()}.png")
        image.save(path)
        imagepaths.append(path)

    return imagepaths


def flush(switti_model, kubin):
    kubin.model.flush(None)
    pipe = switti_model.get("pipe", None)

    if pipe is not None:
        pipe.switti.to("cpu")
        pipe.vae.to("cpu")
        pipe.text_encoder.to("cpu")
        pipe.text_encoder_2.to("cpu")
        switti_model["pipe"] = None

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def mount(kubin):
    sys.path.append(os.path.join("extensions", "kd-switti", "switti"))

    switti_repo = "https://github.com/yandex-research/switti"
    commit = "0f2dbf5"
    destination_dir = "extensions/kd-switti/switti"

    if not os.path.exists(destination_dir):
        current_path = os.getcwd()
        temp_dir = tempfile.mkdtemp()
        temp_switti = os.path.join(temp_dir, "temp_switti")

        subprocess.run(["git", "clone", switti_repo, temp_switti, "-q"])
        os.chdir(temp_switti)
        subprocess.run(["git", "checkout", commit, "-q"])
        os.chdir(current_path)

        if not os.path.exists(destination_dir):
            shutil.copytree(temp_switti, destination_dir)
            init_file_path = os.path.join(destination_dir, "__init__.py")
            open(init_file_path, "a").close()

        for dirpath, dirnames, filenames in os.walk(destination_dir):
            for filename in filenames:
                if filename.endswith(".py"):
                    file_path = os.path.join(dirpath, filename)

                    with open(file_path, "r") as file:
                        content = file.read()

                    modified_content = re.sub(
                        r"^(from\s+)models(\.[a-zA-Z_\.]+\s+import\s+)",
                        r"\1switti.models\2",
                        content,
                        flags=re.MULTILINE,
                    )

                    if modified_content != content:
                        with open(file_path, "w") as file:
                            file.write(modified_content)
                        print(f"Updated imports in {file_path}")

        try:
            shutil.rmtree(temp_dir)
        except:
            pass
