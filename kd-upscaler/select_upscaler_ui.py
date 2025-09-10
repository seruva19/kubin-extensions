import gradio as gr


def upscaler_select_ui(target, create_shared_upscaler_controls_fn):
    with gr.Column() as upscaler_block:
        enable_upscale = gr.Checkbox(
            False,
            label="Enable",
            elem_classes=["inline-flex"],
        )

        upscaler_controls = create_shared_upscaler_controls_fn()

    return upscaler_block
