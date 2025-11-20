from dataclasses import dataclass

import gradio as gr

from ..ui_support import parse_dataset_table


@dataclass
class CachingSection:
    container: gr.Accordion
    cache_dir: gr.Textbox
    cache_enabled: gr.Checkbox
    cache_latents: gr.Checkbox
    cache_text: gr.Checkbox
    cache_overwrite: gr.Checkbox
    cache_target: gr.Dropdown
    cache_status: gr.HTML
    cache_latents_btn: gr.Button
    cache_text_btn: gr.Button
    cache_all_btn: gr.Button


def build_caching_section(default_config, default_dataset_table) -> CachingSection:
    initial_datasets = parse_dataset_table(default_dataset_table)
    initial_choices = ["All"] + [row["name"] for row in initial_datasets]

    with gr.Accordion("Caching", open=True) as container:
        with gr.Row():
            cache_dir = gr.Textbox(
                value=default_config.paths.cache_dir,
                label="Cache root",
            )
            cache_enabled = gr.Checkbox(
                value=default_config.caching.enabled,
                label="Use cached latents",
            )
            cache_latents = gr.Checkbox(
                value=default_config.caching.cache_latents,
                label="Build latents",
            )
            cache_text = gr.Checkbox(
                value=default_config.caching.cache_text,
                label="Build text embeddings",
            )
            cache_overwrite = gr.Checkbox(
                value=default_config.caching.overwrite,
                label="Overwrite cache",
            )

        cache_target = gr.Dropdown(
            choices=initial_choices,
            value="All",
            label="Cache target dataset",
        )
        cache_status = gr.HTML("Cache not built", elem_classes=["kd5-cache-status"])
        with gr.Row():
            cache_latents_btn = gr.Button("Build latents", scale=0)
            cache_text_btn = gr.Button("Build text", scale=0)
            cache_all_btn = gr.Button("Build latents + text", variant="primary")

    container.elem_classes = ["kubin-accordion"]
    return CachingSection(
        container=container,
        cache_dir=cache_dir,
        cache_enabled=cache_enabled,
        cache_latents=cache_latents,
        cache_text=cache_text,
        cache_overwrite=cache_overwrite,
        cache_target=cache_target,
        cache_status=cache_status,
        cache_latents_btn=cache_latents_btn,
        cache_text_btn=cache_text_btn,
        cache_all_btn=cache_all_btn,
    )
