import numpy as np
from skimage import io
from PIL import Image
import gradio as gr
import torch
from RMBG.briarmbg import BriaRMBG
from RMBG.utilities import postprocess_image, preprocess_image
from transformers import AutoModelForImageSegmentation
from torchvision import transforms

title = "BG Remover"

transform_image = transforms.Compose(
    [
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def setup(kubin):
    bg_remover_model_name = None
    bg_remover_model = None

    source_image = gr.Image(
        type="pil",
        label="Image to remove background from",
        elem_classes=["full-height"],
    )

    def remove_background(cache_dir, device, model_name, source_image):
        nonlocal bg_remover_model_name
        nonlocal bg_remover_model

        if bg_remover_model is not None and model_name != bg_remover_model_name:
            bg_remover_model.to("cpu")
            bg_remover_model = None
            torch.cuda.empty_cache()

        if bg_remover_model is None:
            bg_remover_model_name = model_name

            if bg_remover_model_name == "RMBG":
                bg_remover_model = BriaRMBG.from_pretrained(
                    "briaai/RMBG-1.4", cache_dir=cache_dir
                )
                bg_remover_model.to(device)
                bg_remover_model.eval()
            elif bg_remover_model_name == "BiRefNet":
                bg_remover_model = AutoModelForImageSegmentation.from_pretrained(
                    "ZhengPeng7/BiRefNet",
                    trust_remote_code=True,
                    cache_dir=cache_dir,
                )
                bg_remover_model.to(device)
            else:
                raise ValueError(f"Unknown model: {bg_remover_model_name}")

        try:
            img = source_image.convert("RGB")

            if model_name == "RMBG":
                source_image_np = np.array(img)
                model_input_size = [1024, 1024]
                source_image_size = source_image_np.shape[0:2]
                image = (
                    preprocess_image(source_image_np, model_input_size)
                    .to(device)
                    .detach()
                )
                result = bg_remover_model(image)
                result_image = postprocess_image(result[0][0], source_image_size)
                pil_im = Image.fromarray(result_image)
                no_bg_image = Image.new("RGBA", pil_im.size, (0, 0, 0, 0))
                no_bg_image.paste(img, mask=pil_im)
                return no_bg_image

            elif model_name == "BiRefNet":
                image_size = img.size
                input_images = transform_image(img).unsqueeze(0).to(device)

                with torch.no_grad():
                    res = bg_remover_model(input_images)[-1][-1]
                    preds = res.sigmoid().cpu()
                pred = preds[0].squeeze().detach()
                pred_pil = transforms.ToPILImage()(pred)
                mask = pred_pil.resize(image_size)
                transparent_img = img.convert("RGBA")
                transparent_img.putalpha(mask)
                return transparent_img

        except Exception as e:
            print(f"Background removal error with {model_name}: {e}")
            raise

    def bg_remover_ui(ui_shared, ui_tabs):
        with gr.Row() as bg_remover_block:
            with gr.Column(scale=1) as bg_remover_params_block:
                with gr.Row():
                    source_image.render()
                with gr.Row():
                    model = gr.Dropdown(
                        label="Model",
                        choices=["RMBG", "BiRefNet"],
                        value="RMBG",
                    )

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
                    model,
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
