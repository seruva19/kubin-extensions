import gradio as gr
from single_image import interrogate_single_image

title = "Video Tools"


def setup(kubin):
    cache_dir = kubin.params("general", "cache_dir")
    input_video = gr.Video(
        value=None,
        autoplay=False,
        source="upload",
        label="Input video",
    )
    now = {"model": None, "tokenizer": None, "name": None, "fn": None}

    def interrogate(
        model_name, quantization, video, prompt, prepended="", appended="", batch=False
    ):
        if not batch:
            return interrogate_single_image(
                kubin=kubin,
                state=now,
                cache_dir=cache_dir,
                device=kubin.params("general", "device"),
                model_name=model_name,
                quantization=quantization,
                input_video=video,
                input_prompt=prompt,
                prepended=prepended,
                appended=appended,
            )
        else:
            pass

    def video_tools_ui(ui_shared, ui_tabs):
        with gr.Row():
            with gr.Column() as video_interrogator_block:
                with gr.Tab("Interrogation", elem_id="video-interrogation-section"):
                    with gr.Row():
                        with gr.Column():
                            video_model = gr.Dropdown(
                                choices=["THUDM/cogvlm2-video-llama3-chat"],
                                value="THUDM/cogvlm2-video-llama3-chat",
                                label="Model",
                            )

                            model_prompt = gr.Dropdown(
                                allow_custom_value=True,
                                choices=[
                                    "Create a short description of the video",
                                    "Describe this video in details",
                                ],
                                value="Describe this video in details",
                                label="Prompt",
                            )

                            quantization = gr.Dropdown(
                                value="nf4",
                                choices=["none", "int8", "nf4"],
                                label="Quantization",
                            )

            with gr.Column(scale=1):
                with gr.TabItem("Single video"):
                    with gr.Column(scale=1):
                        with gr.Row():
                            input_video.render()

                    with gr.Column(scale=1):
                        interrogate_btn = gr.Button("Interrogate", variant="primary")
                        output_text = gr.Textbox(
                            lines=7,
                            label="Interrogated text",
                            show_copy_button=True,
                        )

                        kubin.ui_utils.click_and_disable(
                            interrogate_btn,
                            fn=interrogate,
                            inputs=[
                                video_model,
                                quantization,
                                input_video,
                                model_prompt,
                            ],
                            outputs=[output_text],
                            js=[
                                f"args => kubin.UI.taskStarted('{title}')",
                                f"args => kubin.UI.taskFinished('{title}')",
                            ],
                        )

        video_interrogator_block.elem_classes = ["block-params"]
        return video_interrogator_block

    return {
        "title": title,
        "send_target": input_video,
        "send_to": f"📄 Send to {title}",
        "tab_ui": lambda ui_s, ts: video_tools_ui(ui_s, ts),
        "send_target": input_video,
    }
