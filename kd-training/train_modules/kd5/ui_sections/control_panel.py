from dataclasses import dataclass
from pathlib import Path

import gradio as gr


@dataclass
class ControlPanel:
    status: gr.HTML
    train_btn: gr.Button
    load_btn: gr.Button
    save_btn: gr.Button
    config_path_input: gr.Textbox


def build_control_panel(default_config_path: Path) -> ControlPanel:
    status = gr.HTML("Training not started", elem_classes=["kd5-training-status"])

    controls_row = gr.Row()
    with controls_row:
        train_btn = gr.Button("Start training", variant="primary")
        load_btn = gr.Button("Load config")
        save_btn = gr.Button("Save config")
        config_path_input = gr.Textbox(
            value=str(default_config_path),
            label="Config file path",
        )

    return ControlPanel(
        status=status,
        train_btn=train_btn,
        load_btn=load_btn,
        save_btn=save_btn,
        config_path_input=config_path_input,
    )
