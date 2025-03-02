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
from torch import nn

title = "Animation"
dir = Path(__file__).parent.absolute()

from kandinsky_video31 import kdv11_create_video


def setup(kubin):
    yaml_config = kubin.yaml_utils.YamlConfig(dir)

    def video_ui(ui_shared, ui_tabs):
        with gr.Tabs(selected=0) as video_block:
            with gr.TabItem("Kandinsky Video 1.0", id=0):
                with gr.Row():
                    with gr.Column(scale=2) as video_params_block:
                        with gr.Row():
                            prompt = gr.TextArea(
                                value="rolling waves on a sandy beach relaxation, rhythm, and coastal beauty, 4k photo",
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
                                50,
                                step=1,
                                label="Steps",
                            )
                            guidance_scale = gr.Slider(
                                1, 30, 5, step=1, label="Guidance scale"
                            )
                            interpolation_guidance_scale = gr.Slider(
                                0,
                                1,
                                0.25,
                                step=0.1,
                                label="Interpolation guidance scale",
                            )
                        with gr.Row():
                            width = gr.Number(value="512", step=8, label="Width")
                            height = gr.Number(value="512", step=8, label="Height")
                            pfps = gr.Dropdown(
                                value="low",
                                choices=["low", "medium", "high"],
                                label="FPS",
                            )

                    with gr.Column(scale=1):
                        create_video_btn = gr.Button("Create video", variant="primary")
                        video_output = gr.Video()

                        kubin.ui_utils.click_and_disable(
                            create_video_btn,
                            fn=lambda *params: create_video(kubin, *params),
                            inputs=[
                                prompt,
                                negative_prompt,
                                width,
                                height,
                                pfps,
                                steps,
                                guidance_scale,
                                interpolation_guidance_scale,
                                gr.Textbox(
                                    value=kubin.params("general", "device"),
                                    visible=False,
                                ),
                                gr.Textbox(
                                    value=kubin.params("general", "cache_dir"),
                                    visible=False,
                                ),
                                gr.Textbox(
                                    value=kubin.params("general", "output_dir"),
                                    visible=False,
                                ),
                                gr.Textbox(
                                    value=kubin.params("native", "text_encoder"),
                                    visible=False,
                                ),
                                gr.State(yaml_config),
                            ],
                            outputs=video_output,
                            js=[
                                f"args => kubin.UI.taskStarted('{title}')",
                                f"args => kubin.UI.taskFinished('{title}')",
                            ],
                        )

                    video_params_block.elem_classes = ["block-params"]

            with gr.TabItem("Kandinsky Video 1.1", id=1):
                with gr.Row():
                    with gr.Column(scale=2) as kdv11_video_params_block:
                        with gr.Row():
                            with gr.Column():
                                with gr.Row():
                                    kdv11_prompt = gr.TextArea(
                                        value="A cat wearing sunglasses and working as a lifeguard at a pool.",
                                        label="Prompt",
                                        lines=3,
                                    )
                                with gr.Row():
                                    kdv11_negative_prompt = gr.TextArea(
                                        value="",
                                        label="Negative prompt",
                                        lines=3,
                                    )
                            with gr.Column():
                                with gr.Row():
                                    kdv11_keyframe_image = gr.Image(
                                        type="pil", label="Keyframe image"
                                    )

                        with gr.Row():
                            kdv11_keyframe_guidance_scale = gr.Slider(
                                1, 30, 5.0, step=0.1, label="Key frame guidance scale"
                            )
                            kdv11_guidance_weight_prompt = gr.Slider(
                                1, 30, 5.0, step=0.1, label="Guidance weight prompt"
                            )
                            kdv11_guidance_weight_image = gr.Slider(
                                1, 30, 3.0, step=0.1, label="Guidance weight image"
                            )
                        with gr.Row():
                            kdv11_interpolation_guidance_scale = gr.Slider(
                                0,
                                1,
                                0.5,
                                step=0.1,
                                label="Interpolation guidance scale",
                            )
                            kdv11_noise_augmentation = gr.Number(
                                20, label="Interpolation guidance scale"
                            )
                            kdv11_fps = gr.Dropdown(
                                value="medium",
                                choices=["low", "medium", "high"],
                                label="FPS",
                            )

                        with gr.Row():
                            kdv11_width = gr.Number(value="512", step=8, label="Width")
                            kdv11_height = gr.Number(
                                value="512", step=8, label="Height"
                            )

                            kdv11_motion = gr.Dropdown(
                                value="high",
                                choices=["low", "medium", "high"],
                                label="Motion",
                            )

                    with gr.Column(scale=1):
                        kdv11_create_video_btn = gr.Button(
                            "Create video", variant="primary"
                        )
                        kdv11_video_output = gr.Gallery()

                        kubin.ui_utils.click_and_disable(
                            kdv11_create_video_btn,
                            fn=lambda *params: kdv11_create_video(kubin, *params),
                            inputs=[
                                kdv11_prompt,
                                kdv11_negative_prompt,
                                kdv11_width,
                                kdv11_height,
                                kdv11_fps,
                                kdv11_motion,
                                kdv11_keyframe_guidance_scale,
                                kdv11_guidance_weight_prompt,
                                kdv11_guidance_weight_image,
                                kdv11_interpolation_guidance_scale,
                                kdv11_noise_augmentation,
                                kdv11_keyframe_image,
                                gr.Textbox(
                                    value=kubin.params("general", "device"),
                                    visible=False,
                                ),
                                gr.Textbox(
                                    value=kubin.params("general", "cache_dir"),
                                    visible=False,
                                ),
                                gr.Textbox(
                                    value=kubin.params("general", "output_dir"),
                                    visible=False,
                                ),
                                gr.Textbox(
                                    value=kubin.params("native", "text_encoder"),
                                    visible=False,
                                ),
                                gr.State(yaml_config),
                            ],
                            outputs=kdv11_video_output,
                            js=[
                                f"args => kubin.UI.taskStarted('{title}')",
                                f"args => kubin.UI.taskFinished('{title}')",
                            ],
                        )

                    kdv11_video_params_block.elem_classes = ["block-params"]
        return video_block

    def settings_ui():
        config = yaml_config.read()

        def save_changes(inputs):
            config["use_lowvram_pipeline"] = inputs[use_lowvram_pipeline]
            yaml_config.write(config)

        with gr.Column() as settings_block:
            use_lowvram_pipeline = gr.Checkbox(
                lambda: config["use_lowvram_pipeline"],
                label="Use low VRAM pipeline",
                scale=0,
            )

            save_btn = gr.Button("Save settings", size="sm", scale=0)
            save_btn.click(
                save_changes,
                inputs={use_lowvram_pipeline},
                queue=False,
            ).then(fn=None, _js=("(x) => kubin.notify.success('Settings saved')"))

        settings_block.elem_classes = ["k-form"]
        return settings_block

    return {
        "title": title,
        "tab_ui": lambda ui_s, ts: video_ui(ui_s, ts),
        "settings_ui": settings_ui,
    }


def create_video(
    kubin,
    prompt,
    negative_prompt,
    width,
    height,
    pfps,
    steps,
    scale,
    int_scale,
    device,
    cache_dir,
    output_dir,
    encoder_path,
    yaml_config,
):
    video_kd3_config = yaml_config.read()
    if video_kd3_config["use_lowvram_pipeline"]:
        kubin.log("using low VRAM t2v pipeline")
        from kandinsky3_video_optimized import get_T2V_pipeline

        t2v_pipe = get_T2V_pipeline(
            device,
            fp16=True,
            fp8=True,
            cache_dir=cache_dir,
            text_encode_path=encoder_path,
        )
    else:
        kubin.log("using original t2v pipeline")

        patch_original_pipeline(kubin)
        from video_kandinsky3 import get_T2V_pipeline

        t2v_pipe = get_T2V_pipeline(
            device, fp16=True, cache_dir=cache_dir, text_encode_path=encoder_path
        )

    video_frames = t2v_pipe(
        text=prompt,
        guidance_scale=scale,
        interpolation_guidance_scale=int_scale,
        negative_text=negative_prompt,
        fps=pfps,
        width=int(width),
        height=int(height),
        steps=steps,
    )

    match pfps:
        case "low":
            fps = 2
        case "medium":
            fps = 8
        case "high":
            fps = 30

    out_video_dir = os.path.join(output_dir, "video")
    video_output_path = os.path.join(out_video_dir, f"video-{uuid4()}.mp4")

    writer = imageio.get_writer(video_output_path, fps=fps)

    for video_frame in video_frames:
        image_array = imageio.core.asarray(video_frame)
        writer.append_data(image_array)
    writer.close()

    return video_output_path


def patch_original_pipeline(kubin):
    from video_kandinsky3.condition_processors import T5TextConditionProcessor
    from video_kandinsky3.condition_encoders import T5TextConditionEncoder
    from video_kandinsky3.utils import freeze

    cache_dir = kubin.params("general", "cache_dir")

    def t5_cond_proc_ctor(self, tokens_length, processor_names):
        self.tokens_length = tokens_length["t5"]
        self.processor = T5Tokenizer.from_pretrained(
            processor_names["t5"], cache_dir=cache_dir
        )

    def t5_cond_enc_ctor(
        self,
        model_names,
        context_dim,
        model_dims,
        low_cpu_mem_usage: bool = True,
        device_map: Optional[str] = None,
    ):
        T5TextConditionEncoder.__base__.__init__(self, context_dim, model_dims)
        t5_model = T5Model.from_pretrained(
            model_names["t5"],
            low_cpu_mem_usage=low_cpu_mem_usage,
            device_map=device_map,
            cache_dir=cache_dir,
        )
        self.encoders = nn.ModuleDict(
            {
                "t5": t5_model.encoder.half(),
            }
        )
        self.encoders = freeze(self.encoders)

    T5TextConditionProcessor.__init__ = t5_cond_proc_ctor
    T5TextConditionEncoder.__init__ = t5_cond_enc_ctor


def mount(kubin):
    mount_kv_10(kubin)
    mount_kv_11(kubin)


def mount_kv_10(kubin):
    kdvideo10_repo = "https://github.com/ai-forever/KandinskyVideo"
    commit = "4e10a23"
    destination_dir = "extensions/kd-video/video_kandinsky3"

    if not os.path.exists(destination_dir):
        current_path = os.getcwd()
        temp_dir = tempfile.mkdtemp()
        temp_kdvideo10 = os.path.join(temp_dir, "temp_kdvideo10")

        subprocess.run(["git", "clone", kdvideo10_repo, temp_kdvideo10, "-q"])
        os.chdir(temp_kdvideo10)
        subprocess.run(["git", "checkout", commit, "-q"])
        os.chdir(current_path)

        repo_path = os.path.join(temp_kdvideo10, "video_kandinsky3")
        if not os.path.exists(destination_dir):
            shutil.copytree(repo_path, destination_dir)
        try:
            shutil.rmtree(temp_dir)
        except:
            pass


def mount_kv_11(kubin):
    kdvideo11_repo = "https://github.com/ai-forever/KandinskyVideo"
    commit = "914eac1"
    destination_dir = "extensions/kd-video/kandinsky_video"

    if not os.path.exists(destination_dir):
        current_path = os.getcwd()
        temp_dir = tempfile.mkdtemp()
        temp_kdvideo11 = os.path.join(temp_dir, "temp_kdvideo11")

        subprocess.run(["git", "clone", kdvideo11_repo, temp_kdvideo11, "-q"])
        os.chdir(temp_kdvideo11)
        subprocess.run(["git", "checkout", commit, "-q"])
        os.chdir(current_path)

        repo_path = os.path.join(temp_kdvideo11, "kandinsky_video")
        if not os.path.exists(destination_dir):
            shutil.copytree(repo_path, destination_dir)
        try:
            shutil.rmtree(temp_dir)
        except:
            pass
