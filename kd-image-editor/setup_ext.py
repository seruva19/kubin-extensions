import gradio as gr

title = "Image Editor"


def setup(kubin):
    input_image = gr.Image(type="numpy", label="Source image")
    input_image.elem_classes = ["kd-image-editor-input"]

    def image_editor_ui(ui_shared, ui_tabs):
        with gr.Column() as ied_block:
            input_image.change(
                fn=None,
                _js="(x) => kubin.imageEditor.openImage(x)",
                inputs=[input_image],
                outputs=None,
            )
            input_image.render()
            gr.HTML('<div id="kd-image-editor-container"></div>')

        return ied_block

    return {
        "send_to": f"🧹 Send to {title}",
        "title": title,
        "tab_ui": lambda ui_s, ts: image_editor_ui(ui_s, ts),
        "send_target": input_image,
    }
