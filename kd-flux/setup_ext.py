import shutil
import subprocess
import tempfile
from uuid import uuid4
import gradio as gr
import os
from pathlib import Path
import torch
from diffusers import FluxTransformer2DModel, FluxPipeline
from transformers import T5EncoderModel, CLIPTextModel
from optimum.quanto import freeze, qfloat8, quantize


title = "BFL Flux"
dir = Path(__file__).parent.absolute()


def setup(kubin):
    from huggingface_hub import constants

    flux_model = {"models": (None, None, None), "model": None}

    def flux_ui(ui_shared, ui_tabs):
        with gr.Row() as flux_block:
            with gr.Column(scale=2) as flux_params_block:
                with gr.Row():
                    prompt = gr.TextArea(
                        value="A photo of a forest with mist swirling around the tree trunks. The word 'FLUX' is painted over it in big, red brush strokes with visible texture",
                        label="Prompt",
                        lines=5,
                    )
                with gr.Row():
                    model_name = gr.Dropdown(
                        value="flux-dev",
                        choices=["flux-dev", "flux-schnell"],
                        label="Model",
                    )
                    width = gr.Slider(
                        128,
                        1360,
                        1360,
                        step=16,
                        label="Width",
                    )
                    height = gr.Slider(
                        128,
                        768,
                        768,
                        step=16,
                        label="Height",
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
                        minimum=1,
                        maximum=20,
                        value=3.5,
                        step=0.5,
                        label="Guidance scale",
                    )
                    seed = gr.Number(-1, label="Seed", precision=0)

                def on_model_change(model):
                    return [
                        gr.update(value=4 if model == "flux-dev" else 50),
                        gr.update(interactive=model == "flux-dev"),
                    ]

                model_name.change(
                    fn=on_model_change,
                    inputs=[model_name],
                    outputs=[steps, guidance_scale],
                )

                with gr.Row():
                    use_offload = gr.Checkbox(True, label="Offload")
                    use_hf_key = gr.Checkbox(True, label="Use local HF token")

                with gr.Row():
                    with gr.Accordion("Utilities", open=False):
                        flush_btn = gr.Button(
                            "Release VRAM", variant="secondary", scale=1
                        )
                        flush_btn.click(
                            fn=lambda: flush_flux(flux_model, kubin),
                            inputs=None,
                            outputs=None,
                        )

            flux_params_block.elem_classes = ["block-params"]

            with gr.Column(scale=1) as flux_output_block:
                generate_btn = gr.Button("Generate", variant="primary")
                flux_output = gr.Gallery(
                    label="Generated Images",
                    columns=2,
                    preview=True,
                    elem_classes=["kd-flux-output"],
                )

                flux_output.select(
                    fn=None,
                    _js=f"() => kubin.UI.setImageIndex('flux-output')",
                    show_progress=False,
                    outputs=gr.State(None),
                )

                ui_shared.create_base_send_targets(flux_output, "flux-output", ui_tabs)
                ui_shared.create_ext_send_targets(flux_output, "flux-output", ui_tabs)

            kubin.ui_utils.click_and_disable(
                generate_btn,
                fn=lambda *params: generate_flux(kubin, flux_model, *params),
                inputs=[
                    model_name,
                    prompt,
                    width,
                    height,
                    steps,
                    guidance_scale,
                    seed,
                    use_offload,
                    use_hf_key,
                    gr.Textbox(value=kubin.params("general", "device"), visible=False),
                    gr.Textbox(
                        value=kubin.params("general", "cache_dir"), visible=False
                    ),
                    gr.Textbox(
                        value=kubin.params("general", "output_dir"), visible=False
                    ),
                ],
                outputs=flux_output,
                js=[
                    f"args => kubin.UI.taskStarted('{title}')",
                    f"args => kubin.UI.taskFinished('{title}')",
                ],
            )

        return flux_block

    return {
        "title": title,
        "tab_ui": lambda ui_s, ts: flux_ui(ui_s, ts),
    }


def generate_flux(
    kubin,
    flux_model,
    model_name,
    prompt,
    width,
    height,
    steps,
    guidance_scale,
    seed,
    use_offload,
    use_hf_key,
    device,
    cache_dir,
    output_dir,
):
    from flux_inference import get_models, text_to_image

    current_model = flux_model.get("model", None)
    if current_model is not None and current_model != model_name:
        flush_flux(flux_model, kubin)

    models = flux_model.get("models", (None, None, None))
    if any(model is None for model in models):
        flux_model["models"] = get_models(
            model_name, device, cache_dir, use_offload, use_hf_key
        )

    pipe, transformer, encoder = flux_model["models"]

    images = text_to_image(
        pipe,
        prompt,
        width,
        height,
        steps,
        guidance_scale,
        seed,
    )

    imagepaths = []
    for image in images:
        folder_path = os.path.join(output_dir, "flux")
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        path = os.path.join(folder_path, f"flux-{uuid4()}.png")
        image.save(path)
        imagepaths.append(path)

    return imagepaths


def flush_flux(flux_model, kubin):
    from flux_inference import flush

    flush(flux_model, kubin)


# def mount(kubin):
#     flux_repo = "https://github.com/black-forest-labs/flux"
#     commit = "eae154e"
#     destination_dir = "extensions/kd-flux/flux"

#     if not os.path.exists(destination_dir):
#         current_path = os.getcwd()
#         temp_dir = tempfile.mkdtemp()
#         temp_flux = os.path.join(temp_dir, "temp_flux")

#         subprocess.run(["git", "clone", flux_repo, temp_flux, "-q"])
#         os.chdir(temp_flux)
#         subprocess.run(["git", "checkout", commit, "-q"])
#         os.chdir(current_path)

#         repo_path = os.path.join(temp_flux, "src", "flux")
#         if not os.path.exists(destination_dir):
#             shutil.copytree(repo_path, destination_dir)
#         try:
#             shutil.rmtree(temp_dir)
#         except:
#             pass
