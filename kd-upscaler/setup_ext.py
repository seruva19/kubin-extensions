import os
import shutil
import subprocess
import tempfile
from upscale import upscale_with
import gradio as gr
from pathlib import Path


dir = Path(__file__).parent.absolute()
default_upscalers_path = f"{dir}/upscalers.default.yaml"

title = "Image Upscaler"


def setup(kubin):
    source_image = gr.Image(
        type="pil", label="Image to upscale", elem_classes=["full-height"]
    )

    def upscaler_ui(ui_shared, ui_tabs):
        with gr.Row() as upscaler_block:
            with gr.Column(scale=1) as upscaler_params_block:
                with gr.Row():
                    source_image.render()

                with gr.Column() as upscale_selector:
                    upscaler = gr.Radio(
                        [
                            "Default (Real-ESRGAN)",
                            "KandiSuperRes",
                            "AuraSR-v2",
                            # "ESRGAN"
                        ],
                        value="Default (Real-ESRGAN)",
                        label="Upscaler",
                    )
                    with gr.Row(visible=True) as default_upscaler_params:
                        scale = gr.Radio(
                            ["2", "4", "8"],
                            value="2",
                            label="Upscale by",
                            interactive=True,
                        )
                    with gr.Row(visible=False) as kdsr_upscaler_params:
                        kdsr_steps = gr.Slider(
                            1,
                            100,
                            5,
                            step=1,
                            label="Steps",
                            elem_classes=["inline-flex"],
                        )
                        kdsr_seed = gr.Number(-1, label="Seed", precision=0)
                        kdsr_batch_size = gr.Slider(
                            1,
                            15,
                            1,
                            step=1,
                            label="View batch size",
                            elem_classes=["inline-flex"],
                        )
                    with gr.Row(visible=False) as aura_upscaler_params:
                        pass

                    def on_method_select(upscaler_method):
                        if upscaler_method == "Default (Real-ESRGAN)":
                            return [
                                gr.update(visible=True),
                                gr.update(visible=False),
                                gr.update(visible=False),
                            ]
                        elif upscaler_method == "KandiSuperRes":
                            return [
                                gr.update(visible=False),
                                gr.update(visible=True),
                                gr.update(visible=False),
                            ]
                        elif upscaler_method == "AuraSR-v2":
                            return [
                                gr.update(visible=False),
                                gr.update(visible=False),
                                gr.update(visible=True),
                            ]

                    upscaler.change(
                        fn=on_method_select,
                        inputs=upscaler,
                        outputs=[
                            default_upscaler_params,
                            kdsr_upscaler_params,
                            aura_upscaler_params,
                        ],
                    )

                with gr.Row():
                    clear_memory = gr.Checkbox(False, label="Clear VRAM before upscale")

            with gr.Column(scale=1):
                upscale_btn = gr.Button("Upscale", variant="primary")
                upscale_output = gr.Gallery(label="Upscaled Image", preview=True)

                upscale_output.select(
                    fn=None,
                    _js=f"() => kubin.UI.setImageIndex('upscale-output')",
                    show_progress=False,
                    outputs=gr.State(None),
                )

                ui_shared.create_base_send_targets(
                    upscale_output, "upscale-output", ui_tabs
                )

            kubin.ui_utils.click_and_disable(
                upscale_btn,
                fn=lambda *p: upscale_with(kubin, *p),
                inputs=[
                    upscaler,
                    gr.Textbox(value=kubin.params("general", "device"), visible=False),
                    gr.Textbox(
                        value=kubin.params("general", "cache_dir"), visible=False
                    ),
                    scale,
                    gr.Textbox(
                        value=kubin.params("general", "output_dir"), visible=False
                    ),
                    source_image,
                    clear_memory,
                    kdsr_steps,
                    kdsr_batch_size,
                    kdsr_seed,
                ],
                outputs=upscale_output,
                js=[
                    f"args => kubin.UI.taskStarted('{title}')",
                    f"args => kubin.UI.taskFinished('{title}')",
                ],
            )

            upscaler_params_block.elem_classes = ["block-params"]
        return upscaler_block

    def upscaler_select_ui(target):
        None

    def upscale_after_inference(target, params, upscale_params):
        None

    return {
        "send_to": f"📐 Send to {title}",
        "title": title,
        "targets": ["t2i", "i2i", "mix", "inpaint", "outpaint"],
        # "inject_ui": lambda target: upscaler_select_ui(target),
        "inject_fn": lambda target, params, augmentations: upscale_after_inference(
            target, params, augmentations[0]
        ),
        "tab_ui": lambda ui_s, ts: upscaler_ui(ui_s, ts),
        "send_target": source_image,
    }


def mount(kubin):
    import fileinput

    kdsr_repo = "https://github.com/ai-forever/KandiSuperRes/"
    commit = "32dc832"
    destination_dir = "extensions/kd-upscaler/KandiSuperRes"

    if not os.path.exists(destination_dir):
        current_path = os.getcwd()
        temp_dir = tempfile.mkdtemp()
        temp_kdsr = os.path.join(temp_dir, "temp_kdsr")

        subprocess.run(["git", "clone", kdsr_repo, temp_kdsr, "-q"])
        os.chdir(temp_kdsr)
        subprocess.run(["git", "checkout", commit, "-q"])
        os.chdir(current_path)

        repo_path = os.path.join(temp_kdsr, "KandiSuperRes")
        if not os.path.exists(destination_dir):
            shutil.copytree(repo_path, destination_dir)
        try:
            shutil.rmtree(temp_dir)
        except:
            pass
