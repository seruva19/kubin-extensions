import gradio as gr
from tabs.interrogate_tab import interrogator_block

title = "Video Tools"


def setup(kubin):
    input_interrogate_video = gr.Video(
        value=None,
        autoplay=False,
        source="upload",
        label="Input video",
    )

    input_scenedetect_video = gr.Video(
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
                    interrogator_block(kubin, title, input_interrogate_video)
        return video_tools_block

    return {
        "title": title,
        "send_target": input_interrogate_video,
        "send_to": f"📄 Send to {title}",
        "tab_ui": lambda ui_s, ts: video_tools_ui(ui_s, ts),
    }
