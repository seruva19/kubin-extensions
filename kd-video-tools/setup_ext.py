import gradio as gr
from tabs.interrogate_tab import interrogator_block
from tabs.video_select_tab import selector_block
from tabs.video_pipeline_tab import pipeline_block
from tabs.video_splitter_tab import splitter_block

title = "Video Tools"


def setup(kubin):
    state = {
        "model": None,
        "tokenizer": None,
        "processor": None,
        "name": None,
        "fn": None,
        "q": None,
    }

    input_interrogate_video = gr.Video(
        value=None,
        autoplay=False,
        source="upload",
        label="Input video",
    )

    input_selector_video = gr.Video(
        value=None,
        autoplay=False,
        source="upload",
        label="Input video",
    )

    def video_tools_ui(ui_shared, ui_tabs):
        with gr.Row() as video_tools_block:
            with gr.Tabs():
                with gr.Tab(
                    "Video Interrogation", elem_id="video-interrogation-section"
                ):
                    interrogator_block(kubin, state, title, input_interrogate_video)
                with gr.Tab("Video Selection", elem_id="video-selection-section"):
                    selector_block(kubin, state, title, input_selector_video)
                with gr.Tab("Video Analyser", elem_id="video-analyser-section"):
                    pipeline_block(kubin, state, title)
                with gr.Tab("Video Splitter", elem_id="video-splitter-section"):
                    splitter_block(kubin, state, title)

        return video_tools_block

    return {
        "title": title,
        "send_target": input_interrogate_video,
        "send_to": f"📄 Send to {title}",
        "tab_ui": lambda ui_s, ts: video_tools_ui(ui_s, ts),
    }
