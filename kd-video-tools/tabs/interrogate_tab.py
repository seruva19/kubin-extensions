import gradio as gr
import os

from apollo_inference import APOLLO_MODEL_ID
from llava_onevision import LLAVA_MODEL_ID
from videochat_flash import VIDEOCHAT_MODEL_ID
from minicpm_v26 import MINICPM_V_MODEL_ID
from minicpm_o26 import MINICPM_OMNI_MODEL_ID
from videollama3 import VIDEOLLAMA3_MODEL_ID
from ovis_16b import OVIS2_MODEL_ID
from qwen_25_vl import QWEN25_VL_MODEL_ID, SHOTVL_MODEL_ID, SKY_CAPTIONER_MODEL_ID
from qwen3_omni import QWEN3_OMNI_MODEL_ID
from qwen25_omni_awq import QWEN25_OMNI_AWQ_MODEL_ID
from qwen3_vl_30b_a3b import QWEN3_VL_30B_A3B_MODEL_ID
from video_r1 import VIDEOR1_MODEL_ID, init_videor1
from gemini_api import init_gemini, GEMINI_MODEL_ID
from keye_vl_8b import KEYE_VL_MODEL_ID
from keye_vl_15 import KEYE_VL_15_MODEL_ID
from avocado_qwen2_5_omni import AVOCADO_MODEL_ID

from functions.video_interrogate import init_interrogate_fn
from functions.classify_video import (
    DEFAULT_CLASSIFIER_TEMPLATE,
    classify,
)

DEFAULT_INTERROGATE_PROMPT = "Create a detailed (short) description of the video"


def interrogator_block(kubin, state, title, input_video):
    cache_dir = kubin.params("general", "cache_dir")

    def interrogate(
        model_name,
        quantization,
        use_flash_attention,
        use_audio_in_video,
        video,
        prompt,
        use_classifier,
        clip_dir,
        clip_types,
        caption_extension,
        overwrite_existing,
        include_subdirectories,
        progress=gr.Progress(),
    ):
        prepended, appended = "", ""

        init_interrogate_fn(
            kubin=kubin,
            state=state,
            cache_dir=cache_dir,
            device=kubin.params("general", "device"),
            model_name=model_name,
            quantization=quantization,
            use_flash_attention=use_flash_attention,
            use_audio_in_video=use_audio_in_video,
        )

        interrogate_fn = state["fn"]

        import inspect

        sig = inspect.signature(interrogate_fn)
        supports_audio_flag = "use_audio_in_video" in sig.parameters

        if clip_dir is None:
            if supports_audio_flag:
                output = interrogate_fn(video, prompt, use_audio_in_video)
            else:
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

            print(f"found {len(relevant_clips)} files to interrogate")
            for image_count, filepath in enumerate(
                progress.tqdm(relevant_clips, unit="files")
            ):
                base_name, _ = os.path.splitext(filepath)

                caption_filename = base_name
                caption_path = f"{caption_filename}{caption_extension}"

                if (
                    not use_classifier
                    and os.path.exists(caption_path)
                    and not overwrite_existing
                ):
                    pass
                else:
                    if supports_audio_flag:
                        output = interrogate_fn(filepath, prompt, use_audio_in_video)
                    else:
                        output = interrogate_fn(filepath, prompt)
                    if use_classifier:
                        classify(filepath, output, clip_dir)
                    else:
                        output = prepended + output + appended

                        with open(caption_path, "w", encoding="utf-8") as file:
                            file.write(output)

            return None, (
                f"{len(relevant_clips)} videos classified"
                if use_classifier
                else f"Captions for {len(relevant_clips)} videos created"
            )

    with gr.Column() as video_interrogator_block:
        with gr.Row():
            video_model = gr.Dropdown(
                choices=[
                    "THUDM/cogvlm2-video-llama3-chat",
                    MINICPM_V_MODEL_ID,
                    MINICPM_OMNI_MODEL_ID,
                    APOLLO_MODEL_ID,
                    LLAVA_MODEL_ID,
                    VIDEOCHAT_MODEL_ID,
                    VIDEOLLAMA3_MODEL_ID,
                    OVIS2_MODEL_ID,
                    QWEN25_VL_MODEL_ID,
                    QWEN3_OMNI_MODEL_ID,
                    QWEN25_OMNI_AWQ_MODEL_ID,
                    AVOCADO_MODEL_ID,
                    QWEN3_VL_30B_A3B_MODEL_ID,
                    SKY_CAPTIONER_MODEL_ID,
                    SHOTVL_MODEL_ID,
                    VIDEOR1_MODEL_ID,
                    KEYE_VL_MODEL_ID,
                    KEYE_VL_15_MODEL_ID,
                    GEMINI_MODEL_ID,
                ],
                value="THUDM/cogvlm2-video-llama3-chat",
                label="Model",
            )
            quantization = gr.Dropdown(
                value="int8",
                choices=["none", "int8", "nf4"],
                label="Quantization",
            )

            with gr.Column():
                use_flash_attention = gr.Checkbox(
                    False,
                    label="Use FlashAttention",
                )
                use_audio_in_video = gr.Checkbox(
                    True,
                    label="Use audio in video",
                )
                activate_classifier = gr.Checkbox(
                    False,
                    label="Activate classifier",
                    info="Only for folder processing",
                )

        with gr.Row():
            model_prompt = gr.TextArea(
                lines=5,
                max_lines=5,
                value=DEFAULT_INTERROGATE_PROMPT,
                label="Prompt",
            )

            activate_classifier.change(
                fn=lambda activate: (
                    DEFAULT_CLASSIFIER_TEMPLATE
                    if activate
                    else DEFAULT_INTERROGATE_PROMPT
                ),
                inputs=[activate_classifier],
                outputs=[model_prompt],
            )

        with gr.Row():
            fake_element = gr.Textbox(visible=False)

            with gr.TabItem("Single video"):
                with gr.Column(scale=1):
                    with gr.Row():
                        input_video.render()

                        with gr.Column(scale=1):
                            interrogate_btn = gr.Button(
                                "Interrogate", variant="primary"
                            )
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
                            use_flash_attention,
                            use_audio_in_video,
                            input_video,
                            model_prompt,
                            activate_classifier,
                            gr.State(None),
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
                clip_dir = gr.Textbox(label="Directory with source files")

                with gr.Row():
                    clip_types = gr.CheckboxGroup(
                        [".mp4", ".jpg", ".png"],
                        value=[".mp4"],
                        label="Files to interrogate",
                        info="If specific VLM does not have native image interrogation capability, then images will be treated as single-frame videos",
                    )

                with gr.Row():
                    caption_extension = gr.Textbox(
                        value=".txt",
                        info="Caption files extension",
                        label="Caption files",
                    )
                    with gr.Column():
                        overwrite_existing = gr.Checkbox(
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
                        use_flash_attention,
                        use_audio_in_video,
                        input_video,
                        model_prompt,
                        activate_classifier,
                        clip_dir,
                        clip_types,
                        caption_extension,
                        overwrite_existing,
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
