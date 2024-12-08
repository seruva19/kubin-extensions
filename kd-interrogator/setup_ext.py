import gradio as gr
from PIL import Image
from clip_interrogator import Config, Interrogator
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    AutoModelForCausalLM,
    BlipForConditionalGeneration,
    Blip2ForConditionalGeneration,
)
import re
import pandas as pd
import torch
import os


title = "Interrogator"

from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports


def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    if not str(filename).endswith("modeling_florence2.py"):
        return get_imports(filename)
    imports = get_imports(filename)
    imports.remove("flash_attn")
    return imports


def setup(kubin):
    cache_dir = kubin.params("general", "cache_dir")

    CAPTION_MODELS = {
        "blip-base": "Salesforce/blip-image-captioning-base",
        "blip-large": "Salesforce/blip-image-captioning-large",
        "blip2-2.7b": "Salesforce/blip2-opt-2.7b",
        "blip2-flan-t5-xl": "Salesforce/blip2-flan-t5-xl",
        "git-large-coco": "microsoft/git-large-coco",
    }

    def patched_load_caption_model(self):
        if self.config.caption_model is None and self.config.caption_model_name:
            if not self.config.quiet:
                print(f"Loading caption model {self.config.caption_model_name}...")

            model_path = CAPTION_MODELS[self.config.caption_model_name]
            if self.config.caption_model_name.startswith("git-"):
                caption_model = AutoModelForCausalLM.from_pretrained(
                    model_path, torch_dtype=torch.float32, cache_dir=cache_dir
                )
            elif self.config.caption_model_name.startswith("blip2-"):
                caption_model = Blip2ForConditionalGeneration.from_pretrained(
                    model_path, torch_dtype=self.dtype, cache_dir=cache_dir
                )
            else:
                caption_model = BlipForConditionalGeneration.from_pretrained(
                    model_path, torch_dtype=self.dtype, cache_dir=cache_dir
                )
            self.caption_processor = AutoProcessor.from_pretrained(
                model_path, cache_dir=cache_dir
            )

            caption_model.eval()
            if not self.config.caption_offload:
                caption_model = caption_model.to(self.config.device)
            self.caption_model = caption_model
        else:
            self.caption_model = self.config.caption_model
            self.caption_processor = self.config.caption_processor

    Interrogator.load_caption_model = patched_load_caption_model

    ci = None
    ci_config = None
    source_image = gr.Image(type="pil", label="Input image", elem_classes=[])

    def get_interrogator(clip_model, blip_type, cache_path, chunk_size):
        nonlocal ci
        nonlocal ci_config

        if ci is None or [clip_model, blip_type] != ci_config:
            ci_config = [clip_model, blip_type]
            ci = Interrogator(
                Config(
                    clip_model_name=clip_model,
                    caption_model_name=blip_type,
                    clip_model_path=cache_path,
                    cache_path=cache_path,
                    download_cache=True,
                    chunk_size=chunk_size,
                )
            )

        return ci

    vlm_model = None
    vlm_model_id = ""
    vlm_model_fn = lambda _: "Cannot find relevant model"

    def get_vlm_interrogator_fn(model_id, quantization):
        nonlocal vlm_model
        nonlocal vlm_model_id
        nonlocal vlm_model_fn

        device = "cuda"
        dtype = torch.float16 if device == "cuda" else torch.float32

        if vlm_model is None or vlm_model_id != model_id:
            print(f"initializing {model_id} for interrogation")
            vlm_model_id = model_id

            if vlm_model_id == "vikhyatk/moondream2":
                revision = "2024-07-23"
                vlm_model = AutoModelForCausalLM.from_pretrained(
                    vlm_model_id,
                    trust_remote_code=True,
                    revision=revision,
                    cache_dir=cache_dir,
                ).to(device)

                tokenizer = AutoTokenizer.from_pretrained(
                    vlm_model_id, revision=revision, cache_dir=cache_dir
                )

                def answer(image, model, tokenizer, prompt):
                    enc_image = model.encode_image(image)
                    return model.answer_question(enc_image, prompt, tokenizer)

                vlm_model_fn = lambda i, p: answer(
                    image=i, model=vlm_model, tokenizer=tokenizer, prompt=p
                )

            elif vlm_model_id == "microsoft/Florence-2-large":
                vlm_model = AutoModelForCausalLM.from_pretrained(
                    "microsoft/Florence-2-large",
                    cache_dir=cache_dir,
                    torch_dtype=dtype,
                    trust_remote_code=True,
                ).to(device)

                processor = AutoProcessor.from_pretrained(
                    "microsoft/Florence-2-large",
                    cache_dir=cache_dir,
                    trust_remote_code=True,
                )

                def answer(image, vision_model, processor, prompt):
                    inputs = processor(
                        text=prompt, images=image, return_tensors="pt"
                    ).to(device, dtype)

                    generated_ids = vision_model.generate(
                        input_ids=inputs["input_ids"],
                        pixel_values=inputs["pixel_values"],
                        max_new_tokens=1024,
                        early_stopping=False,
                        num_beams=3,
                        do_sample=False,
                    )
                    generated_text = processor.batch_decode(
                        generated_ids, skip_special_tokens=False
                    )[0]
                    parsed_answer = processor.post_process_generation(
                        generated_text,
                        task=prompt,
                        image_size=(image.width, image.height),
                    )
                    return parsed_answer

                vlm_model_fn = lambda i, p: answer(
                    image=i,
                    vision_model=vlm_model,
                    processor=processor,
                    prompt=p,
                )[p]

            elif vlm_model_id == "fancyfeast/joy-caption-pre-alpha":
                from models.joy_caption_pre import JoyCaptionPreAlphaInterrogatorModel

                vlm_model = JoyCaptionPreAlphaInterrogatorModel()
                vlm_model.load_components(cache_dir, device)
                vlm_model_fn = lambda i, p: vlm_model.get_caption(i, p)

            elif vlm_model_id == "fancyfeast/joy-caption-alpha-one":
                from models.joy_caption_alpha import JoyCaptionAlphaOneInterrogatorModel

                vlm_model = JoyCaptionAlphaOneInterrogatorModel()
                vlm_model.load_components(cache_dir, device)
                vlm_model_fn = lambda i, p: vlm_model.get_caption(i, p)

            elif vlm_model_id == "fancyfeast/joy-caption-alpha-two":
                from models.joy_caption_alpha_two import (
                    JoyCaptionAlphaTwoInterrogatorModel,
                )

                vlm_model = JoyCaptionAlphaTwoInterrogatorModel()
                vlm_model.load_components(cache_dir, device)
                vlm_model_fn = lambda i, p: vlm_model.get_caption(i, p)

            elif vlm_model_id == "THUDM/cogvlm2-llama3-chat-19B":
                from models.cogvlm2 import CogVLM2Model

                vlm_model = CogVLM2Model()
                vlm_model.load_model(cache_dir, device, quantization)
                vlm_model_fn = lambda i, p: vlm_model.get_caption(i, p)

            elif vlm_model_id == "internlm/internlm-xcomposer2-4khd-7b":
                from models.internlm_xc2_4khd import InternLM2Model

                vlm_model = InternLM2Model()
                vlm_model.load_model(cache_dir, device, quantization)
                vlm_model_fn = lambda i, p: vlm_model.get_caption(i, p)

            elif vlm_model_id == "Qwen/Qwen2-VL-7B-Instruct":
                from models.qwen2_vl import Qwen2VLModel

                vlm_model = Qwen2VLModel()
                vlm_model.load_model(cache_dir, device)
                vlm_model_fn = lambda i, p: vlm_model.get_caption(i, p)

            elif vlm_model_id == "Ertugrul/Qwen2-VL-7B-Captioner-Relaxed":
                from models.qwen2_vl_relaxed import Qwen2VLRelaxedModel

                vlm_model = Qwen2VLRelaxedModel()
                vlm_model.load_model(cache_dir, device)
                vlm_model_fn = lambda i, p: vlm_model.get_caption(i, p)

            elif vlm_model_id == "Ertugrul/Pixtral-12B-Captioner-Relaxed":
                from models.pixtral_relaxed import PixtralRelaxedModel

                vlm_model = PixtralRelaxedModel()
                vlm_model.load_model(cache_dir, device, quantization)
                vlm_model_fn = lambda i, p: vlm_model.get_caption(i, p)

            elif vlm_model_id == "openbmb/MiniCPM-V-2_6":
                from models.mini_cpm import MiniCPMModel

                vlm_model = MiniCPMModel()
                vlm_model.load_model(cache_dir, device)
                vlm_model_fn = lambda i, p: vlm_model.get_caption(i, p)

            elif vlm_model_id == "cyan2k/molmo-7B-O-bnb-4bit":
                from models.molmo import Molmo7BModel

                vlm_model = Molmo7BModel()
                vlm_model.load_model(cache_dir, device)
                vlm_model_fn = lambda i, p: vlm_model.get_caption(i, p)

        return vlm_model_fn

    def route_interrogate(
        model_index,
        image,
        mode,
        clip_model,
        blip_type,
        chunk_size,
        vlm_model,
        vlm_prompt,
        prepended_txt,
        appended_txt,
        quantization,
    ):
        if model_index == 0:
            return interrogate(
                image,
                mode,
                clip_model,
                blip_type,
                chunk_size,
                prepended_txt,
                appended_txt,
            )
        elif model_index == 1:
            return vlm_interrogate(
                image, vlm_model, vlm_prompt, prepended_txt, appended_txt, quantization
            )

    def interrogate(
        image, mode, clip_model, blip_type, chunk_size, prepended_txt, appended_txt
    ):
        image = image.convert("RGB")
        interrogated_text = ""

        interrogator = get_interrogator(
            clip_model=clip_model,
            blip_type=blip_type,
            cache_path=f"{cache_dir}/clip_cache",
            chunk_size=chunk_size,
        )
        if mode == "best":
            interrogated_text = interrogator.interrogate(image)
        elif mode == "classic":
            interrogated_text = interrogator.interrogate_classic(image)
        elif mode == "fast":
            interrogated_text = interrogator.interrogate_fast(image)
        elif mode == "negative":
            interrogated_text = interrogator.interrogate_negative(image)

        return prepended_txt + interrogated_text + appended_txt

    def vlm_interrogate(
        image, model, prompt, prepended_txt, appended_txt, quantization
    ):
        image = image.convert("RGB")
        vlm_interrogator_fn = get_vlm_interrogator_fn(
            model_id=model, quantization=quantization
        )
        interrogated_text = vlm_interrogator_fn(image, prompt)
        return prepended_txt + interrogated_text + appended_txt

    def batch_interrogate(
        model_index,
        image_dir,
        batch_mode,
        image_extensions,
        skip_existing,
        remove_line_breaks,
        output_dir,
        caption_extension,
        output_csv,
        mode,
        clip_model,
        blip_type,
        chunk_size,
        vlm_model,
        vlm_prompt,
        prepended_txt,
        appended_txt,
        quantization,
        progress=gr.Progress(),
    ):
        if output_dir == "":
            output_dir = image_dir
        os.makedirs(output_dir, exist_ok=True)

        relevant_images = []

        progress(0, desc="Starting batch interrogation...")
        if not os.path.exists(image_dir):
            return f"Error: image folder {image_dir} does not exists"

        for filename in os.listdir(image_dir):
            if filename.endswith(tuple(image_extensions)):
                relevant_images.append([filename, f"{image_dir}/{filename}", ""])

        print(f"found {len(relevant_images)} images to interrogate")
        image_count = 0
        for _ in progress.tqdm(relevant_images, unit="images"):
            filename = relevant_images[image_count][0]
            filepath = relevant_images[image_count][1]

            caption_filename = os.path.splitext(filename)[0]
            caption_path = f"{output_dir}/{caption_filename}{caption_extension}"

            if skip_existing and os.path.exists(caption_path) and batch_mode == 0:
                with open(caption_path, "r", encoding="utf-8") as file:
                    caption = file.read()
                if batch_mode == 1:
                    relevant_images[image_count][2] = caption
            else:
                image = Image.open(filepath)
                if model_index == 0:
                    caption = interrogate(
                        image,
                        mode,
                        clip_model,
                        blip_type,
                        chunk_size,
                        prepended_txt,
                        appended_txt,
                    )
                elif model_index == 1:
                    caption = vlm_interrogate(
                        image,
                        vlm_model,
                        vlm_prompt,
                        prepended_txt,
                        appended_txt,
                        quantization,
                    )

                if remove_line_breaks:
                    caption = re.sub(r"[\r\n]+", " ", caption)

                if batch_mode == 0:
                    with open(caption_path, "w", encoding="utf-8") as file:
                        file.write(caption)
                elif batch_mode == 1:
                    relevant_images[image_count][2] = caption

            image_count += 1

        if batch_mode == 1:
            captions_df = pd.DataFrame(
                [i[1:] for i in relevant_images], columns=["image_name", "caption"]
            )
            csv_path = f"{output_dir}/{output_csv}"
            captions_df.to_csv(csv_path, index=False)
            print(f"CSV file with captions saved to {csv_path}")

        return f"Captions for {len(relevant_images)} images created"

    def interrogator_ui(ui_shared, ui_tabs):
        with gr.Row() as interrogator_block:
            with gr.Column(scale=1) as interrogator_params_block:
                with gr.Tabs() as interrogator_panels:
                    with gr.Tab("CLIP"):
                        with gr.Row():
                            clip_model = gr.Dropdown(
                                choices=[
                                    "ViT-L-14/openai",
                                    "ViT-H-14/laion2b_s32b_b79k",
                                ],
                                value="ViT-L-14/openai",
                                label="CLIP model",
                            )
                        with gr.Row():
                            mode = gr.Radio(
                                ["best", "classic", "fast", "negative"],
                                value="fast",
                                label="Mode",
                            )
                        with gr.Row():
                            blip_model_type = gr.Radio(
                                ["blip-base", "blip-large", "git-large-coco"],
                                value="blip-large",
                                label="Caption model name",
                            )
                        with gr.Row():
                            chunk_size = gr.Slider(
                                512, 2048, 2048, step=512, label="Chunk size"
                            )
                    with gr.Tab("VLM"):
                        with gr.Row() as vlm_interrogator_block:
                            with gr.Column(scale=1) as vlm_interrogator_params_block:
                                with gr.Row():
                                    vlm_model = gr.Dropdown(
                                        choices=[
                                            "vikhyatk/moondream2",
                                            "microsoft/Florence-2-large",
                                            "THUDM/cogvlm2-llama3-chat-19B",
                                            "internlm/internlm-xcomposer2-4khd-7b",
                                            "Qwen/Qwen2-VL-7B-Instruct",
                                            "Ertugrul/Qwen2-VL-7B-Captioner-Relaxed",
                                            "Ertugrul/Pixtral-12B-Captioner-Relaxed",
                                            "openbmb/MiniCPM-V-2_6",
                                            "fancyfeast/joy-caption-pre-alpha",
                                            "fancyfeast/joy-caption-alpha-one",
                                            "fancyfeast/joy-caption-alpha-two",
                                            "cyan2k/molmo-7B-O-bnb-4bit",
                                        ],
                                        value="vikhyatk/moondream2",
                                        label="VLM name",
                                    )
                                with gr.Row():
                                    vlm_prompt = gr.Textbox(
                                        "Output the detailed description of this image.",
                                        label="System prompt",
                                        lines=3,
                                        max_lines=3,
                                    )

                                with gr.Row() as additional_params:
                                    quantization = gr.Dropdown(
                                        value="4bit",
                                        choices=["None", "8bit", "4bit"],
                                        label="Quantization",
                                        scale=1,
                                    )
                                    batch_size = gr.Number(
                                        value=1,
                                        minimum=1,
                                        maximum=128,
                                        step=1,
                                        label="Batch size",
                                        interactive=False,
                                        scale=1,
                                    )

                                def change_vlm_prompt(vlm_model):
                                    prompt = (
                                        "Describe this image as detailed as possible."
                                    )
                                    if vlm_model == "vikhyatk/moondream2":
                                        prompt = "Output the detailed description of this image."
                                    elif vlm_model == "microsoft/Florence-2-large":
                                        prompt = "<MORE_DETAILED_CAPTION>"
                                    elif vlm_model == "THUDM/cogvlm2-llama3-chat-19B":
                                        prompt = "Describe the image."
                                    elif (
                                        vlm_model
                                        == "internlm/internlm-xcomposer2-4khd-7b"
                                    ):
                                        prompt = "<ImageHere>Please describe this image in detail."
                                    elif vlm_model == "Qwen/Qwen2-VL-7B-Instruct":
                                        prompt = "Describe this image as detailed as possible, even the slightest details should be preserved."
                                    elif (
                                        vlm_model
                                        == "Ertugrul/Qwen2-VL-7B-Captioner-Relaxed"
                                    ):
                                        prompt = "Provide a highly detailed and objective description of the image. Describe only observable elements without speculation, ensuring no detail, no matter how small, is overlooked."
                                    elif (
                                        vlm_model
                                        == "Ertugrul/Pixtral-12B-Captioner-Relaxed"
                                    ):
                                        prompt = "Describe the image.\n"

                                    elif vlm_model == "openbmb/MiniCPM-V-2_6":
                                        prompt = (
                                            "Make a detailed description of this image."
                                        )
                                    elif (
                                        vlm_model == "fancyfeast/joy-caption-pre-alpha"
                                    ):
                                        prompt = (
                                            "A descriptive caption for this image:\n"
                                        )
                                    elif (
                                        vlm_model == "fancyfeast/joy-caption-alpha-one"
                                    ):
                                        prompt = "descriptive,formal,very long"
                                    elif (
                                        vlm_model == "fancyfeast/joy-caption-alpha-two"
                                    ):
                                        prompt = "Descriptive,very long"

                                    elif vlm_model == "cyan2k/molmo-7B-O-bnb-4bit":
                                        prompt = "Describe this image."

                                    return prompt

                                vlm_model.change(
                                    fn=change_vlm_prompt,
                                    inputs=[vlm_model],
                                    outputs=[vlm_prompt],
                                )
                            vlm_interrogator_params_block.elem_classes = [
                                "block-params"
                            ]
                model_index = gr.State(0)

                def on_tabs_select(evt: gr.SelectData):
                    return evt.index

                interrogator_panels.select(on_tabs_select, None, model_index)

                with gr.Row() as extra_params_block:
                    prepend_text = gr.Textbox(
                        "",
                        label="Text to prepend caption with",
                        lines=2,
                        max_lines=2,
                    )
                    append_text = gr.Textbox(
                        "",
                        label="Text to append to caption",
                        lines=2,
                        max_lines=2,
                    )

            with gr.Column(scale=1):
                with gr.Tabs():
                    with gr.TabItem("Single image"):
                        with gr.Column(scale=1):
                            with gr.Row():
                                source_image.render()

                        with gr.Column(scale=1):
                            interrogate_btn = gr.Button(
                                "Interrogate", variant="primary"
                            )
                            target_text = gr.Textbox(
                                lines=5,
                                label="Interrogated text",
                                show_copy_button=True,
                            )

                            kubin.ui_utils.click_and_disable(
                                interrogate_btn,
                                fn=route_interrogate,
                                inputs=[
                                    model_index,
                                    source_image,
                                    mode,
                                    clip_model,
                                    blip_model_type,
                                    chunk_size,
                                    vlm_model,
                                    vlm_prompt,
                                    prepend_text,
                                    append_text,
                                    quantization,
                                ],
                                outputs=[target_text],
                                js=[
                                    f"args => kubin.UI.taskStarted('{title}')",
                                    f"args => kubin.UI.taskFinished('{title}')",
                                ],
                            )
                    with gr.TabItem("Batch"):
                        image_dir = gr.Textbox(label="Directory with images")

                        with gr.Row():
                            image_types = gr.CheckboxGroup(
                                [".jpg", ".jpeg", ".png", ".bmp"],
                                value=[".jpg", ".jpeg", ".png", ".bmp"],
                                label="Files to interrogate",
                            )

                        with gr.Row():
                            caption_mode = gr.Radio(
                                choices=["text files", "csv dataset"],
                                info="Save captions to separate text files or to a single csv file",
                                value="text files",
                                label="Caption save mode",
                                type="index",
                            )
                            with gr.Column():
                                skip_existing = gr.Checkbox(
                                    True,
                                    label="Skip if caption exists",
                                )
                                remove_line_breaks = gr.Checkbox(
                                    False,
                                    label="Remove line breaks",
                                )

                        output_dir = gr.Textbox(
                            label="Output folder",
                            info="If empty, the same folder will be used",
                        )

                        caption_extension = gr.Textbox(
                            ".txt",
                            label="Caption files extension",
                            visible=True,
                        )

                        output_csv = gr.Textbox(
                            value="captions.csv",
                            label="Name of csv file",
                            visible=False,
                        )

                        caption_mode.select(
                            fn=lambda m: [
                                gr.update(visible=m != 0),
                                gr.update(visible=m == 0),
                            ],
                            inputs=[caption_mode],
                            outputs=[caption_extension, output_csv],
                        )

                        batch_interrogate_btn = gr.Button(
                            "Interrogate", variant="primary"
                        )
                        progress = gr.HTML(
                            label="Interrogation progress",
                            elem_classes=["batch-interrogation-progress"],
                        )

                        kubin.ui_utils.click_and_disable(
                            batch_interrogate_btn,
                            fn=batch_interrogate,
                            inputs=[
                                model_index,
                                image_dir,
                                caption_mode,
                                image_types,
                                skip_existing,
                                remove_line_breaks,
                                output_dir,
                                caption_extension,
                                output_csv,
                                mode,
                                clip_model,
                                blip_model_type,
                                chunk_size,
                                vlm_model,
                                vlm_prompt,
                                prepend_text,
                                append_text,
                                quantization,
                            ],
                            outputs=[progress],
                            js=[
                                f"args => kubin.UI.taskStarted('{title}')",
                                f"args => kubin.UI.taskFinished('{title}')",
                            ],
                        )

            interrogator_params_block.elem_classes = ["block-params"]
        return interrogator_block

    return {
        "title": title,
        "send_to": f"📄 Send to {title}",
        "tab_ui": lambda ui_s, ts: interrogator_ui(ui_s, ts),
        "send_target": source_image,
        "api": {
            interrogate: lambda image, mode="fast", clip_model="ViT-L-14/openai", blip_type="large", chunks=2048: interrogate(
                image, mode, clip_model, blip_type, chunks
            )
        },
    }
