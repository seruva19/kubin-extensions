from dataclasses import dataclass

import gradio as gr

from train_modules.train_tools import text_tip


@dataclass
class PreviewSection:
    container: gr.Accordion
    preview_enabled: gr.Checkbox
    preview_step: gr.Number
    preview_epoch: gr.Number
    preview_dir: gr.Textbox
    preview_prompts: gr.Dataframe
    preview_gallery: gr.Gallery
    refresh_button: gr.Button
    preview_info: gr.HTML


def build_preview_section(default_config, default_preview_rows) -> PreviewSection:
    with gr.Accordion("Preview Renders", open=True) as container:
        with gr.Row():
            preview_enabled = gr.Checkbox(
                value=default_config.previews.enabled,
                label="Enable previews",
                info="Generate sample videos during training to monitor progress",
            )
            preview_step = gr.Number(
                value=default_config.previews.step_interval,
                label="Preview every N steps",
                precision=0,
                info="Generate previews every N training steps (0 to disable)",
            )
            preview_epoch = gr.Number(
                value=default_config.previews.epoch_interval,
                label="Preview every N epochs",
                precision=0,
                info="Generate previews every N epochs (0 to disable)",
            )

        preview_dir = gr.Textbox(
            value=default_config.previews.get("output_dir", ""),
            label="Preview output directory",
            info=text_tip("Defaults to <output>/previews"),
        )
        # Include headers as the first row for Gradio 3.50.2
        headers = [
            "Prompt",
            "Negative",
            "Time (s)",
            "Seed",
            "Width",
            "Height",
            "Bucket",
        ]
        preview_prompts_value = (
            [headers] + default_preview_rows if default_preview_rows else [headers]
        )

        preview_prompts = gr.Dataframe(
            value=preview_prompts_value,
            datatype=["str", "str", "number", "number", "number", "number", "str"],
            row_count=(len(default_preview_rows) + 2, "dynamic"),
            wrap=True,
        )
        gr.Markdown(
            "Use the optional `Bucket` column to match training aspect buckets (e.g. `768x512`). Leave blank to rely on explicit width/height."
        )
        preview_gallery = gr.Gallery(
            label="Preview renders",
            columns=2,
            height="auto",
            allow_preview=True,
        )
        with gr.Row():
            refresh_button = gr.Button("Refresh previews", size="sm", scale=0)
            preview_info = gr.HTML("", elem_classes=["kd5-preview-info"])

    container.elem_classes = ["kubin-accordion"]
    return PreviewSection(
        container=container,
        preview_enabled=preview_enabled,
        preview_step=preview_step,
        preview_epoch=preview_epoch,
        preview_dir=preview_dir,
        preview_prompts=preview_prompts,
        preview_gallery=preview_gallery,
        refresh_button=refresh_button,
        preview_info=preview_info,
    )
