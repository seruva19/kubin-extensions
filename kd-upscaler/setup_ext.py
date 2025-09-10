import os
import shutil
import subprocess
import tempfile
from upscale import upscale_with
from upscaler_esrgan_user import get_custom_models_list, CustomESRGANUpscaler
from select_upscaler_ui import upscaler_select_ui
import gradio as gr
from pathlib import Path


dir = Path(__file__).parent.absolute()
default_upscalers_path = f"{dir}/upscalers.default.yaml"

title = "Image Upscaler"


def get_model_path_from_selection(custom_model_selection):
    if not custom_model_selection:
        return None

    model_name = custom_model_selection.split(" (")[0]

    custom_models = get_custom_models_list()
    for model in custom_models:
        if model["name"] == model_name:
            return model["path"]

    return None


def create_shared_upscaler_controls():
    custom_models = get_custom_models_list()

    upscaler_options = ["Default (Real-ESRGAN)", "KandiSuperRes", "AuraSR-v2", "HYPIR", "ESRGAN (user)"]

    upscaler = gr.Radio(
        upscaler_options,
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

    with gr.Row(visible=False) as hypir_upscaler_params:
        hypir_prompt = gr.Textbox(
            value="",
            label="Enhancement Prompt",
            placeholder="Optional: Describe what you want to enhance (e.g., 'sharp details', 'vibrant colors')",
            lines=4,
            max_lines=4,
        )
        hypir_scale = gr.Slider(
            1,
            8,
            2,
            step=1,
            label="Upscale Factor",
            elem_classes=["inline-flex"],
        )
        hypir_seed = gr.Number(-1, label="Seed", precision=0)

    with gr.Row(visible=False) as esrgan_user_params:
        custom_model_dropdown = gr.Dropdown(
            interactive=True,
            choices=(
                [f"{model['name']} ({model['scale']}x)" for model in custom_models]
                if custom_models
                else ["No models available - click Refresh after adding models"]
            ),
            value=(
                custom_models[0]["name"] + f" ({custom_models[0]['scale']}x)"
                if custom_models
                else "No models available - click Refresh after adding models"
            ),
            label="ESRGAN model",
            info="Select an ESRGAN model from the available options",
        )

        refresh_models_btn = gr.Button("🔄 Refresh Models", size="sm")

        if not custom_models:
            gr.Markdown(
                "**No custom models found.** \n\n"
                "Place your ESRGAN-compatible .pth files in `models/upscaler/` folder. "
                "Then click 'Refresh Models' to load them."
            )

    def on_method_select(upscaler_method):
        if upscaler_method == "Default (Real-ESRGAN)":
            return [
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
            ]
        elif upscaler_method == "KandiSuperRes":
            return [
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
            ]
        elif upscaler_method == "AuraSR-v2":
            return [
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
            ]
        elif upscaler_method == "HYPIR":
            return [
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),
            ]
        elif upscaler_method == "ESRGAN (user)":
            return [
                gr.update(visible=False),
                gr.update(visible=False),
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
            hypir_upscaler_params,
            esrgan_user_params,
        ],
    )

    clear_memory = gr.Checkbox(False, label="Clear VRAM before upscale")

    def refresh_custom_models():
        updated_models = get_custom_models_list()
        choices = (
            [f"{model['name']} ({model['scale']}x)" for model in updated_models]
            if updated_models
            else ["No models available - click Refresh after adding models"]
        )
        value = choices[0] if choices else "No models available - click Refresh after adding models"
        return gr.update(choices=choices, value=value)

    refresh_models_btn.click(fn=refresh_custom_models, outputs=custom_model_dropdown)

    return {
        "upscaler": upscaler,
        "scale": scale,
        "kdsr_steps": kdsr_steps,
        "kdsr_seed": kdsr_seed,
        "kdsr_batch_size": kdsr_batch_size,
        "custom_model_dropdown": custom_model_dropdown,
        "clear_memory": clear_memory,
        "hypir_prompt": hypir_prompt,
        "hypir_scale": hypir_scale,
        "hypir_seed": hypir_seed,
        "param_groups": {
            "default": default_upscaler_params,
            "kdsr": kdsr_upscaler_params,
            "aura": aura_upscaler_params,
            "hypir": hypir_upscaler_params,
            "custom": esrgan_user_params,
        },
    }


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
                    upscaler_controls = create_shared_upscaler_controls()

            with gr.Column(scale=1):
                upscale_btn = gr.Button("Upscale", variant="primary")
                upscale_output = gr.Gallery(label="Upscaled Image", preview=True)

                upscale_output.select(
                    fn=None,
                    _js=f"() => kubin.UI.setImageIndex('upscale-output')",
                    show_progress="hidden",
                    outputs=gr.State(None),
                )

                ui_shared.create_base_send_targets(
                    upscale_output, "upscale-output", ui_tabs
                )

            def upscale_wrapper(
                upscaler_method,
                device,
                cache_dir,
                scale,
                output_dir,
                input_image,
                clear_memory,
                kdsr_steps,
                kdsr_batch_size,
                kdsr_seed,
                custom_model_selection,
                hypir_prompt,
                hypir_scale,
                hypir_seed,
            ):
                model_path = None
                if upscaler_method == "ESRGAN (user)":
                    if not custom_model_selection or "No models available" in custom_model_selection:
                        return [
                            {
                                "error": "No ESRGAN models available. Please add .pth files to models/upscaler/ folder and click 'Refresh Models'."
                            }
                        ]
                    model_path = get_model_path_from_selection(custom_model_selection)
                    if not model_path:
                        return [
                            {
                                "error": "Selected ESRGAN model file not found. Please check if the model file exists and click 'Refresh Models'."
                            }
                        ]

                if upscaler_method == "HYPIR":
                    return upscale_with(
                        kubin,
                        upscaler_method,
                        device,
                        cache_dir,
                        hypir_scale,
                        output_dir,
                        input_image,
                        clear_memory,
                        kdsr_steps,
                        kdsr_batch_size,
                        hypir_seed,
                        model_path,
                        hypir_prompt,
                    )

                return upscale_with(
                    kubin,
                    upscaler_method,
                    device,
                    cache_dir,
                    scale,
                    output_dir,
                    input_image,
                    clear_memory,
                    kdsr_steps,
                    kdsr_batch_size,
                    kdsr_seed,
                    model_path,
                )

            kubin.ui_utils.click_and_disable(
                upscale_btn,
                fn=upscale_wrapper,
                inputs=[
                    upscaler_controls["upscaler"],
                    gr.Textbox(value=kubin.params("general", "device"), visible=False),
                    gr.Textbox(
                        value=kubin.params("general", "cache_dir"), visible=False
                    ),
                    upscaler_controls["scale"],
                    gr.Textbox(
                        value=kubin.params("general", "output_dir"), visible=False
                    ),
                    source_image,
                    upscaler_controls["clear_memory"],
                    upscaler_controls["kdsr_steps"],
                    upscaler_controls["kdsr_batch_size"],
                    upscaler_controls["kdsr_seed"],
                    upscaler_controls["custom_model_dropdown"],
                    upscaler_controls["hypir_prompt"],
                    upscaler_controls["hypir_scale"],
                    upscaler_controls["hypir_seed"],
                ],
                outputs=upscale_output,
                js=[
                    f"args => kubin.UI.taskStarted('{title}')",
                    f"args => kubin.UI.taskFinished('{title}')",
                ],
            )

            upscaler_params_block.elem_classes = ["block-params"]
        return upscaler_block

    def upscale_after_inference(target, params, upscale_params, *args):
        print(upscale_params)

    def on_hook(hook, **kwargs):
        if hook == kubin.params.HOOK.PIPELINE_IMAGE_GENERATED:
            image = kwargs["image"]
            params = kwargs["params"]
            task = kwargs["task"]
        return None

    return {
        "send_to": f"📐 Send to {title}",
        "title": title,
        "targets": ["t2i", "i2i", "mix", "inpaint", "outpaint"],
        "inject_ui": lambda target: upscaler_select_ui(
            target, create_shared_upscaler_controls
        ),
        "inject_fn": lambda target, params, augmentations: upscale_after_inference(
            kubin, target, params, *augmentations
        ),
        "tab_ui": lambda ui_s, ts: upscaler_ui(ui_s, ts),
        "send_target": source_image,
        "hook_fn": on_hook,
        "inject_position": "before_params",
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

    hypir_repo = "https://github.com/XPixelGroup/HYPIR/"
    hypir_destination_dir = "extensions/kd-upscaler/HYPIR"

    if not os.path.exists(hypir_destination_dir):
        current_path = os.getcwd()
        temp_dir = tempfile.mkdtemp()
        temp_hypir = os.path.join(temp_dir, "temp_hypir")

        try:
            subprocess.run(["git", "clone", hypir_repo, temp_hypir, "-q"], check=True)
            os.chdir(temp_hypir)
            os.chdir(current_path)

            hypir_source_path = os.path.join(temp_hypir, "HYPIR")
            if os.path.exists(hypir_source_path):
                shutil.copytree(hypir_source_path, hypir_destination_dir)
                print(f"HYPIR files copied to {hypir_destination_dir}")
            else:
                print(f"Warning: HYPIR subdirectory not found in repository")

        except subprocess.CalledProcessError as e:
            print(f"Error cloning HYPIR repository: {e}")
        except Exception as e:
            print(f"Error setting up HYPIR: {e}")
        finally:
            try:
                shutil.rmtree(temp_dir)
            except:
                pass
