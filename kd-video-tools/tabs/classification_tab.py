import gradio as gr
import os


def classification_block(kubin, title, input_video):
    cache_dir = kubin.params("general", "cache_dir")

    with gr.Column() as video_classification_block:
        with gr.Row():
            pass

    video_classification_block.elem_classes = ["block-params"]
    return video_classification_block
