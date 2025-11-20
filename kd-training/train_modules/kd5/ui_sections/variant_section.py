from dataclasses import dataclass

import gradio as gr

from train_modules.train_tools import text_tip

from ..ui_support import VARIANT_CHOICES, config_path_for_variant


@dataclass
class VariantSection:
    container: gr.Accordion
    variant: gr.Dropdown
    config_path: gr.components.Textbox


def build_variant_section(default_config) -> VariantSection:
    default_variant = default_config.variant.name
    default_config_path = default_config.variant.get(
        "config", config_path_for_variant(default_variant)
    )

    with gr.Accordion("Config", open=True) as container:
        with gr.Row():
            variant = gr.Dropdown(
                choices=VARIANT_CHOICES,
                value=default_variant,
                label="Model variant",
            )
            config_path = gr.Textbox(
                value=default_config_path,
                label="Pipeline config path",
            )

    container.elem_classes = ["kubin-accordion"]
    return VariantSection(container=container, variant=variant, config_path=config_path)
