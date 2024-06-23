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
        "blip-base": "Salesforce/blip-image-captioning-base",  # 990MB
        "blip-large": "Salesforce/blip-image-captioning-large",  # 1.9GB
        "blip2-2.7b": "Salesforce/blip2-opt-2.7b",  # 15.5GB
        "blip2-flan-t5-xl": "Salesforce/blip2-flan-t5-xl",  # 15.77GB
        "git-large-coco": "microsoft/git-large-coco",  # 1.58GB
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

    def get_vlm_interrogator_fn(model_id, model_prompt):
        nonlocal vlm_model
        nonlocal vlm_model_id
        nonlocal vlm_model_fn

        if vlm_model is None or vlm_model_id != model_id:
            print(f"initializing {model_id} for interrogation")
            vlm_model_id = model_id
            if vlm_model_id == "vikhyatk/moondream2":
                revision = "2024-05-20"
                vlm_model = AutoModelForCausalLM.from_pretrained(
                    vlm_model_id,
                    trust_remote_code=True,
                    revision=revision,
                    cache_dir=cache_dir,
                )
                tokenizer = AutoTokenizer.from_pretrained(
                    vlm_model_id, revision=revision, cache_dir=cache_dir
                )

                def answer(image, model, tokenizer, prompt):
                    enc_image = model.encode_image(image)
                    return model.answer_question(enc_image, prompt, tokenizer)

                vlm_model_fn = lambda i: answer(
                    image=i, model=vlm_model, tokenizer=tokenizer, prompt=model_prompt
                )
            elif vlm_model_id == "microsoft/Florence-2-large":
                with patch(
                    "transformers.dynamic_module_utils.get_imports", fixed_get_imports
                ):
                    vision_model = AutoModelForCausalLM.from_pretrained(
                        "microsoft/Florence-2-large",
                        cache_dir=cache_dir,
                        trust_remote_code=True,
                    )
                processor = AutoProcessor.from_pretrained(
                    "microsoft/Florence-2-large",
                    cache_dir=cache_dir,
                    trust_remote_code=True,
                )

                def answer(image, vision_model, processor, prompt):
                    inputs = processor(text=prompt, images=image, return_tensors="pt")

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

                vlm_model_fn = lambda i: answer(
                    image=i,
                    vision_model=vision_model,
                    processor=processor,
                    prompt=model_prompt,
                )[model_prompt]
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
    ):
        if model_index == 0:
            return interrogate(image, mode, clip_model, blip_type, chunk_size)
        elif model_index == 1:
            return vlm_interrogate(image, vlm_model, vlm_prompt)

    def interrogate(image, mode, clip_model, blip_type, chunk_size):
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

        return interrogated_text

    def vlm_interrogate(image, model, prompt):
        image = image.convert("RGB")
        vlm_interrogator_fn = get_vlm_interrogator_fn(
            model_id=model,
            model_prompt=prompt,
        )
        interrogated_text = vlm_interrogator_fn(image)
        return interrogated_text

    def batch_interrogate(
        model_index,
        image_dir,
        batch_mode,
        image_extensions,
        output_dir,
        caption_extension,
        output_csv,
        mode,
        clip_model,
        blip_type,
        chunk_size,
        vlm_model,
        vlm_prompt,
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

            image = Image.open(filepath)
            if model_index == 0:
                caption = interrogate(image, mode, clip_model, blip_type, chunk_size)
            elif model_index == 1:
                caption = vlm_interrogate(image, vlm_model, vlm_prompt)

            if batch_mode == 0:
                caption_filename = os.path.splitext(filename)[0]
                with open(
                    f"{output_dir}/{caption_filename}{caption_extension}",
                    "w",
                    encoding="utf-8",
                ) as file:
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

                                def change_vlm_prompt(vlm_model):
                                    if vlm_model == "vikhyatk/moondream2":
                                        return "Output the detailed description of this image."
                                    elif vlm_model == "microsoft/Florence-2-large":
                                        return "<MORE_DETAILED_CAPTION>"

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

                        caption_mode = gr.Radio(
                            choices=["text files", "csv dataset"],
                            info="Save captions to separate text files or to a single csv file",
                            value="text files",
                            label="Caption save mode",
                            type="index",
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
                        progress = gr.HTML(label="Interrogation progress")

                        kubin.ui_utils.click_and_disable(
                            batch_interrogate_btn,
                            fn=batch_interrogate,
                            inputs=[
                                model_index,
                                image_dir,
                                caption_mode,
                                image_types,
                                output_dir,
                                caption_extension,
                                output_csv,
                                mode,
                                clip_model,
                                blip_model_type,
                                chunk_size,
                                vlm_model,
                                vlm_prompt,
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
