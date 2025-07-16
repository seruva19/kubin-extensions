import gradio as gr
import os
import json
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from tqdm import tqdm

from functions.video_interrogate import init_interrogate_fn
from ovis_16b import OVIS2_MODEL_ID
from qwen_25_vl import QWEN25_VL_MODEL_ID
from functions.video_selector import (
    STYLE_ANALYSIS_PROMPT,
    VideoSelector,
    create_analysis_prompt,
)


def selector_block(kubin, state, title, input_video):
    selector = VideoSelector(kubin, state)

    with gr.Blocks(title="Video selector", theme=gr.themes.Soft()) as selector_ui:
        with gr.Tab("Video analysis"):
            with gr.Row():
                video_model = gr.Dropdown(
                    choices=[
                        OVIS2_MODEL_ID,
                        QWEN25_VL_MODEL_ID,
                    ],
                    value=QWEN25_VL_MODEL_ID,
                    label="VLM",
                )
                quantization = gr.Dropdown(
                    value="int8",
                    choices=["none", "int8", "nf4"],
                    label="Quantization",
                )

            with gr.Row():
                use_flash_attention = gr.Checkbox(
                    False,
                    label="Use FlashAttention",
                )
                include_subdirectories = gr.Checkbox(
                    False,
                    label="Include subdirectories",
                )

            with gr.Row():
                model_prompt = gr.TextArea(
                    lines=5,
                    max_lines=5,
                    value=create_analysis_prompt(STYLE_ANALYSIS_PROMPT, ""),
                    label="Prompt",
                )

                video_dir = gr.Textbox(
                    label="Video Directory",
                    placeholder="Path to directory containing videos",
                )
                file_extensions = gr.CheckboxGroup(
                    choices=[".mp4", ".avi", ".mov", ".mkv", ".webm"],
                    value=[".mp4"],
                    label="Video file extensions",
                )

            analyze_btn = gr.Button("Analyze", variant="primary")

            analysis_output = gr.Textbox(label="Results", lines=8, interactive=False)

            analysis_dataframe = gr.Dataframe(label="Videos", interactive=False)

        with gr.Tab("Video Selection"):
            with gr.Row():
                num_videos_to_select = gr.Number(
                    value=10, minimum=1, maximum=100, label="Number of videos to select"
                )
                select_btn = gr.Button("Select best videos", variant="primary")

            selection_output = gr.Textbox(
                label="Selection results", lines=10, interactive=False
            )

            selection_dataframe = gr.Dataframe(
                label="Selected videos", interactive=False
            )

        with gr.Tab("Export"):
            with gr.Row():
                export_filename = gr.Textbox(
                    value="selected_videos.json", label="Export filename"
                )
                export_btn = gr.Button("Export Selected Videos", variant="secondary")

            export_output = gr.Textbox(
                label="Export Status", lines=3, interactive=False
            )

        kubin.ui_utils.click_and_disable(
            analyze_btn,
            fn=selector.analyze_videos,
            inputs=[
                model_prompt,
                video_dir,
                video_model,
                quantization,
                use_flash_attention,
                file_extensions,
                include_subdirectories,
            ],
            outputs=[analysis_output, analysis_dataframe],
            js=[
                f"args => kubin.UI.taskStarted('{title}')",
                f"args => kubin.UI.taskFinished('{title}')",
            ],
        )

        kubin.ui_utils.click_and_disable(
            select_btn,
            fn=selector.select_best_videos,
            inputs=[num_videos_to_select],
            outputs=[selection_output, selection_dataframe],
            js=[
                f"args => kubin.UI.taskStarted('{title}')",
                f"args => kubin.UI.taskFinished('{title}')",
            ],
        )

        kubin.ui_utils.click_and_disable(
            export_btn,
            fn=selector.export_selected_videos,
            inputs=[export_filename],
            outputs=[export_output],
            js=[
                f"args => kubin.UI.taskStarted('{title}')",
                f"args => kubin.UI.taskFinished('{title}')",
            ],
        )

    return selector_ui
