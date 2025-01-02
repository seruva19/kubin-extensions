import os
import gradio as gr
from video_interrogate import init_interrogate_fn
from apollo_inference import APOLLO_MODEL_ID

title = "Video Tools"


def setup(kubin):
    cache_dir = kubin.params("general", "cache_dir")
    input_video = gr.Video(
        value=None,
        autoplay=False,
        source="upload",
        label="Input video",
    )
    now = {"model": None, "tokenizer": None, "name": None, "fn": None, "q": None}

    def interrogate(
        model_name,
        quantization,
        video,
        prompt,
        clip_dir,
        clip_types,
        caption_extension,
        skip_existing,
        include_subdirectories,
        progress=gr.Progress(),
    ):
        prepended, appended = "", ""

        init_interrogate_fn(
            kubin=kubin,
            state=now,
            cache_dir=cache_dir,
            device=kubin.params("general", "device"),
            model_name=model_name,
            quantization=quantization,
        )

        interrogate_fn = now["fn"]

        if clip_dir is None:
            output = interrogate_fn(video, prompt)
            return prepended + output + appended, None
        else:
            relevant_clips = []

            progress(0, desc="Starting folder interrogation...")
            if not os.path.exists(clip_dir):
                return f"Error: folder {clip_dir} does not exists"

            if include_subdirectories:
                for root, dirnames, filenames in os.walk(clip_dir):
                    for filename in filenames:
                        if filename.endswith(tuple(clip_types)):
                            relevant_clips.append(f"{root}/{filename}")
            else:
                for filename in os.listdir(clip_dir):
                    if filename.endswith(tuple(clip_types)):
                        relevant_clips.append(f"{clip_dir}/{filename}")

            print(f"found {len(relevant_clips)} images to interrogate")
            for image_count, filepath in enumerate(
                progress.tqdm(relevant_clips, unit="videos")
            ):
                base_name, _ = os.path.splitext(filepath)

                caption_filename = base_name
                caption_path = f"{caption_filename}{caption_extension}"

                if os.path.exists(caption_path) and skip_existing:
                    pass
                else:
                    output = interrogate_fn(filepath, prompt)
                    output = prepended + output + appended

                    with open(caption_path, "w", encoding="utf-8") as file:
                        file.write(output)

            return None, f"Captions for {len(relevant_clips)} videos created"

    def video_tools_ui(ui_shared, ui_tabs):
        with gr.Row():
            with gr.Column() as video_interrogator_block:
                with gr.Tab("Interrogation", elem_id="video-interrogation-section"):
                    with gr.Row():
                        with gr.Column():
                            video_model = gr.Dropdown(
                                choices=[
                                    "THUDM/cogvlm2-video-llama3-chat",
                                    APOLLO_MODEL_ID,
                                ],
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
                                value="int8",
                                choices=["none", "int8", "nf4"],
                                label="Quantization",
                            )

            with gr.Column(scale=1):
                fake_element = gr.Textbox(visible=False)

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
                                gr.State(None),
                                gr.State(None),
                                gr.State(None),
                                gr.State(None),
                            ],
                            outputs=[output_text, fake_element],
                            js=[
                                f"args => kubin.UI.taskStarted('{title}')",
                                f"args => kubin.UI.taskFinished('{title}')",
                            ],
                        )

                with gr.TabItem("Folder"):
                    clip_dir = gr.Textbox(label="Directory with videos")

                    with gr.Row():
                        clip_types = gr.CheckboxGroup(
                            [".mp4", ".jpg", ".png"],
                            value=[".mp4"],
                            label="Files to interrogate",
                            info="Images will be treated as single-frame videos",
                        )

                    with gr.Row():
                        caption_extension = gr.Radio(
                            choices=[".txt"],
                            info="Target captions",
                            value=".txt",
                            label="Caption files",
                        )
                        with gr.Column():
                            skip_existing = gr.Checkbox(
                                False,
                                label="Overwrite existing",
                            )
                            include_subdirectories = gr.Checkbox(
                                False,
                                label="Include all subdirectories",
                            )

                    folder_interrogate_btn = gr.Button("Interrogate", variant="primary")
                    progress = gr.HTML(
                        label="Interrogation progress",
                        elem_classes=["folder-interrogation-progress"],
                    )

                    kubin.ui_utils.click_and_disable(
                        folder_interrogate_btn,
                        fn=interrogate,
                        inputs=[
                            video_model,
                            quantization,
                            input_video,
                            model_prompt,
                            clip_dir,
                            clip_types,
                            caption_extension,
                            skip_existing,
                            include_subdirectories,
                        ],
                        outputs=[fake_element, progress],
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
