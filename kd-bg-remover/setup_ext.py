import numpy as np
from skimage import io
from PIL import Image
import gradio as gr
from RMBG.briarmbg import BriaRMBG
from RMBG.utilities import postprocess_image, preprocess_image

title = "BG Remover"


def setup(kubin):
    rmbg = None

    source_image = gr.Image(
        type="pil",
        label="Image to remove background from",
        elem_classes=["full-height"],
    )

    def remove_background(cache_dir, device, source_image):
        nonlocal rmbg

        if rmbg is None:
            rmbg = BriaRMBG()
            rmbg = BriaRMBG.from_pretrained("briaai/RMBG-1.4", cache_dir=cache_dir)
            rmbg.to(device)
            rmbg.eval()

        source_image_np = np.array(source_image)
        model_input_size = [1024, 1024]
        source_image_size = source_image_np.shape[0:2]
        image = preprocess_image(source_image_np, model_input_size).to(device)
        result = rmbg(image)
        result_image = postprocess_image(result[0][0], source_image_size)

        pil_im = Image.fromarray(result_image)
        no_bg_image = Image.new("RGBA", pil_im.size, (0, 0, 0, 0))
        no_bg_image.paste(source_image, mask=pil_im)

        return no_bg_image

    def bg_remover_ui(ui_shared, ui_tabs):
        with gr.Row() as bg_remover_block:
            with gr.Column(scale=1) as bg_remover_params_block:
                with gr.Row():
                    source_image.render()

            with gr.Column(scale=1):
                remove_bg_btn = gr.Button(
                    "Remove background", label="Remove background", variant="primary"
                )
                no_bg_output = gr.Image(label="Image without background")

            kubin.ui_utils.click_and_disable(
                remove_bg_btn,
                fn=remove_background,
                inputs=[
                    gr.State(kubin.params("general", "cache_dir")),
                    gr.State(kubin.params("general", "device")),
                    source_image,
                ],
                outputs=no_bg_output,
                js=[
                    f"args => kubin.UI.taskStarted('{title}')",
                    f"args => kubin.UI.taskFinished('{title}')",
                ],
            )

            bg_remover_params_block.elem_classes = ["block-params"]
        return bg_remover_block

    return {
        "title": title,
        "send_to": f"✂️ Send to {title}",
        "tab_ui": lambda ui_s, ts: bg_remover_ui(ui_s, ts),
        "send_target": source_image,
    }
