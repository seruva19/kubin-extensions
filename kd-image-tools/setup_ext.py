from PIL import Image
import os
import gradio as gr
from similarity_search import ImageSimilaritySearch
import pandas as pd
import shutil
import torch
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn

from utils.logging import k_log

title = "Image Tools"


def setup(kubin):
    cache_dir = kubin.params("general", "cache_dir")

    def image_tools_ui(ui_shared, ui_tabs):
        current_folder = {"path": ""}

        def group_similar_files(
            folder_path: str, top_results: int, score_threshold: float
        ):
            try:
                if not os.path.exists(folder_path):
                    k_log(f"folder not found: {folder_path}")
                    return "Folder not found"

                search_engine = ImageSimilaritySearch()
                results_df = search_engine.find_similar_images_batch(
                    folder_path, top_results, score_threshold
                )

                if not isinstance(results_df, pd.DataFrame):
                    results_df = pd.DataFrame(results_df)

                grouped_dir = os.path.join(folder_path, "grouped_similar")
                os.makedirs(grouped_dir, exist_ok=True)

                processed_images = set()
                group_counter = 1

                for _, row in results_df.iterrows():
                    query_img = row["Query Image"]
                    similar_img = row["Similar Image"]

                    if (
                        query_img in processed_images
                        and similar_img in processed_images
                    ):
                        continue

                    if query_img not in processed_images:
                        group_folder = os.path.join(
                            grouped_dir, f"similar_group_{group_counter}"
                        )
                        os.makedirs(group_folder, exist_ok=True)

                        query_path = os.path.join(folder_path, query_img)
                        if os.path.exists(query_path):
                            shutil.move(
                                query_path, os.path.join(group_folder, query_img)
                            )
                            processed_images.add(query_img)
                            k_log(f"Moved {query_img} to group {group_counter}")

                        similar_path = os.path.join(folder_path, similar_img)
                        if (
                            os.path.exists(similar_path)
                            and similar_img not in processed_images
                        ):
                            shutil.move(
                                similar_path, os.path.join(group_folder, similar_img)
                            )
                            processed_images.add(similar_img)
                            k_log(f"Moved {similar_img} to group {group_counter}")

                        group_counter += 1

                ungrouped_dir = os.path.join(grouped_dir, "ungrouped")
                os.makedirs(ungrouped_dir, exist_ok=True)

                valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
                for filename in os.listdir(folder_path):
                    if (
                        os.path.splitext(filename)[1].lower() in valid_exts
                        and filename not in processed_images
                    ):
                        file_path = os.path.join(folder_path, filename)
                        if os.path.isfile(file_path):
                            shutil.move(
                                file_path, os.path.join(ungrouped_dir, filename)
                            )
                            k_log(f"Moved {filename} to ungrouped folder")

                return f"Successfully moved similar images into {group_counter-1} groups, remaining images moved to 'ungrouped' folder"
            except Exception as e:
                error_msg = f"Error in group_similar_files: {str(e)}"
                k_log(error_msg)
                return error_msg

        def on_image_select(image_name, results_df):
            if results_df is None or results_df.empty or not image_name:
                return None, None, None, None

            try:
                if not isinstance(results_df, pd.DataFrame):
                    results_df = pd.DataFrame(results_df)

                image_name = (
                    os.path.basename(image_name)
                    if os.path.sep in image_name
                    else image_name
                )
                similar_images = results_df[
                    results_df["Query Image"].astype(str) == image_name
                ]

                query_path = os.path.join(current_folder["path"], image_name)

                if not os.path.exists(query_path):
                    k_log(f"query image not found: {query_path}")
                    return None, None, None, None

                query_image = Image.open(query_path)
                width, height = query_image.size
                source_image_info = f"Resolution: {width}x{height}, Path: {image_name}"

                return (query_image, query_path, similar_images, source_image_info)
            except Exception as e:
                k_log(f"error in on_image_select: {str(e)}")
                return None, None, None, None

        def process_folder(folder_path: str, top_results: int, score_threshold: float):
            try:
                if not os.path.exists(folder_path):
                    k_log(f"folder not found: {folder_path}")
                    return [pd.DataFrame(), gr.update(choices=[], value=None)]

                current_folder["path"] = folder_path

                search_engine = ImageSimilaritySearch()
                results_df = search_engine.find_similar_images_batch(
                    folder_path, top_results, score_threshold
                )

                if not isinstance(results_df, pd.DataFrame):
                    results_df = pd.DataFrame(results_df)

                unique_images = sorted(results_df["Query Image"].unique().tolist())

                image_choices = [(img, img) for img in unique_images]

                print(f"{len(unique_images)} images added for review")
                return [
                    results_df,
                    gr.update(choices=image_choices, value=None),
                    gr.update(visible=True),
                    gr.update(visible=True),
                    len(results_df),
                ]
            except Exception as e:
                print(f"Error in process_folder: {str(e)}")
                return [
                    pd.DataFrame(),
                    gr.update(choices=[], value=None),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    0,
                ]

        def mark_as_resolved(image_name, results_df):
            if not image_name or results_df is None or results_df.empty:
                return results_df, None, []

            new_df = results_df[
                (results_df["Query Image"] != image_name)
                & (results_df["Similar Image"] != image_name)
            ]

            unique_images = sorted(new_df["Query Image"].unique().tolist())
            image_choices = [(img, img) for img in unique_images]
            return new_df, gr.update(choices=image_choices, value=None), []

        def remove_source_image(image_name, results_df, remove_permanently):
            if not image_name or results_df is None or results_df.empty:
                return results_df, None, []

            try:
                full_path = os.path.join(current_folder["path"], image_name)

                if not remove_permanently:
                    duplicates_folder = os.path.join(
                        current_folder["path"], "duplicates"
                    )
                    os.makedirs(duplicates_folder, exist_ok=True)
                    try:
                        new_duplicate_path = os.path.join(duplicates_folder, image_name)
                        shutil.move(full_path, new_duplicate_path)
                        k_log(f"moved source image to: {new_duplicate_path}")
                    except Exception as e:
                        k_log(f"error moving source image {full_path}: {str(e)}")
                else:
                    try:
                        if os.path.exists(full_path):
                            os.remove(full_path)
                            k_log(f"deleted source image: {full_path}")
                    except Exception as e:
                        k_log(f"error deleting source image {full_path}: {str(e)}")

                # Remove this image from results DataFrame
                new_df = results_df[
                    (results_df["Query Image"] != image_name)
                    & (results_df["Similar Image"] != image_name)
                ]

                unique_images = sorted(new_df["Query Image"].unique().tolist())
                image_choices = [(img, img) for img in unique_images]
                return new_df, gr.update(choices=image_choices, value=None), []

            except Exception as e:
                k_log(f"Error removing source image: {str(e)}")
                return results_df, None, []

        def remove_similar_image(
            selected_gallery_item,
            image_name,
            results_df,
            remove_permanently,
            current_gallery,
        ):
            if (
                not selected_gallery_item
                or results_df is None
                or results_df.empty
                or not image_name
            ):
                return results_df, current_gallery

            try:
                selected_item = selected_gallery_item[0][1]
                image_filename = selected_item.split("Path: ")[-1].strip()

                if not image_filename:
                    return results_df, current_gallery

                full_path = os.path.join(current_folder["path"], image_filename)

                if not remove_permanently:
                    duplicates_folder = os.path.join(
                        current_folder["path"], "duplicates"
                    )
                    os.makedirs(duplicates_folder, exist_ok=True)
                    try:
                        new_duplicate_path = os.path.join(
                            duplicates_folder, image_filename
                        )
                        shutil.move(full_path, new_duplicate_path)
                        k_log(f"moved duplicate image to: {new_duplicate_path}")
                    except Exception as e:
                        k_log(f"error moving image {full_path}: {str(e)}")
                else:
                    try:
                        if os.path.exists(full_path):
                            os.remove(full_path)
                            k_log(f"deleted image: {full_path}")
                    except Exception as e:
                        k_log(f"error deleting image {full_path}: {str(e)}")

                # Remove the image from results DataFrame
                results_df = results_df[
                    (results_df["Similar Image"] != image_filename)
                    | (results_df["Query Image"] != image_name)
                ]

                # Update gallery by removing the selected image
                updated_gallery = []
                for img, caption in current_gallery:
                    if image_filename not in caption:
                        updated_gallery.append((img, caption))

                return results_df, updated_gallery
            except Exception as e:
                k_log(f"Error in remove_similar_image: {str(e)}")
                return results_df, current_gallery

        def update_display(image_name, results_df):
            selected_img, path, similar_df, source_image_info = on_image_select(
                image_name, results_df
            )

            if selected_img is None:
                return [None, "", [], None]

            gallery_images = []
            if similar_df is not None and not similar_df.empty:
                for _, data in similar_df.iterrows():
                    full_path = os.path.join(
                        current_folder["path"], data["Similar Image"]
                    )
                    try:
                        img = Image.open(full_path)
                        width, height = img.size  # Get image resolution
                        gallery_images.append(
                            (
                                img,
                                f"Score: {data['Similarity Score']}, Rank: {data['Rank']}, "
                                f"Resolution: {width}x{height}, Path: {data['Similar Image']}",
                            )
                        )
                    except Exception as e:
                        k_log(f"Error loading image {full_path}: {str(e)}")

            return [selected_img, source_image_info, gallery_images, path]

        with gr.Tabs() as image_tools_block:
            with gr.Tab("Similarity search"):
                with gr.Blocks():
                    with gr.Row():
                        folder_input = gr.Text(
                            scale=4,
                            label="Image folder path",
                            placeholder="Enter the path to your image folder",
                            lines=1,
                            elem_classes="kd-similarity-folder-path",
                        )
                        with gr.Column():
                            top_images = gr.Slider(
                                scale=2,
                                minimum=1,
                                maximum=20,
                                value=10,
                                step=1,
                                label="Number of similar images to find per image",
                            )
                            score_threshold = gr.Slider(
                                scale=2,
                                minimum=0,
                                maximum=1,
                                value=0.95,
                                step=0.05,
                                label="Score threshold to exclude similar images",
                            )
                        with gr.Column():
                            remove_duplicates = gr.Checkbox(
                                False,
                                label="Remove duplicates permanently",
                            )

                        with gr.Column():
                            process_btn = gr.Button("Process Images")
                            group_btn = gr.Button("Group into folders")

                    grouping_result = gr.Text(
                        label="Grouping Result", interactive=False, visible=True
                    )
                    kubin.ui_utils.click_and_disable(
                        group_btn,
                        fn=group_similar_files,
                        inputs=[folder_input, top_images, score_threshold],
                        outputs=[grouping_result],
                        js=[
                            f"args => kubin.UI.taskStarted('{title}')",
                            f"args => kubin.UI.taskFinished('{title}')",
                        ],
                    )

                    results_store = gr.State()
                    selected_image_path = gr.State()
                    gallery_state = gr.State([])

                    with gr.Row(visible=False) as similar_images:
                        with gr.Column(scale=1) as available_images:
                            image_selector = gr.Radio(
                                choices=[],
                                label="All images in folder",
                                value=None,
                                interactive=True,
                                elem_classes="kd-similarity-scrollable-list",
                            )

                        available_images.elem_classes = [
                            "kd-similarity-available-images"
                        ]

                        with gr.Column(scale=3):
                            with gr.Row():
                                with gr.Column():
                                    selected_image = gr.Image(
                                        type="pil", label="Source image"
                                    )
                                    selected_image_info = gr.Label(
                                        "", label="Image info"
                                    )
                                    with gr.Row():
                                        resolve_btn = gr.Button(
                                            "Mark as Resolved", variant="primary"
                                        )
                                        remove_source_btn = gr.Button(
                                            "Remove Source Image", variant="stop"
                                        )

                                with gr.Column():
                                    similar_images_gallery = gr.Gallery(
                                        label="Similar images",
                                        show_label=True,
                                        columns=4,
                                        elem_classes=["kd-similarity-similar-images"],
                                    )
                                    remove_btn = gr.Button(
                                        "Remove Selected Similar Image",
                                        variant="secondary",
                                    )

                        similar_images.elem_classes = ["kd-similarity-similar-images"]

                    result_length = gr.Number(-1, visible=False)

                    kubin.ui_utils.click_and_disable(
                        process_btn,
                        fn=process_folder,
                        inputs=[folder_input, top_images, score_threshold],
                        outputs=[
                            results_store,
                            image_selector,
                            similar_images,
                            folder_input,
                            result_length,
                        ],
                        js=[
                            f"args => kubin.UI.taskStarted('{title}')",
                            f"args => kubin.UI.taskFinished('{title}')",
                        ],
                    ).then(
                        fn=None,
                        _js='(n) => (n == 0 && kubin.notify.error("Found 0 similar images!"), n)',
                        inputs=[result_length],
                        outputs=[],
                    )

                    image_selector.change(
                        fn=update_display,
                        inputs=[image_selector, results_store],
                        outputs=[
                            selected_image,
                            selected_image_info,
                            similar_images_gallery,
                            selected_image_path,
                        ],
                    ).then(
                        fn=lambda x: x,
                        inputs=[similar_images_gallery],
                        outputs=[gallery_state],
                    )

                    resolve_btn.click(
                        fn=mark_as_resolved,
                        inputs=[image_selector, results_store],
                        outputs=[results_store, image_selector, similar_images_gallery],
                    )

                    remove_source_btn.click(
                        fn=remove_source_image,
                        inputs=[image_selector, results_store, remove_duplicates],
                        outputs=[results_store, image_selector, similar_images_gallery],
                    )

                    remove_btn.click(
                        fn=remove_similar_image,
                        inputs=[
                            similar_images_gallery,
                            image_selector,
                            results_store,
                            remove_duplicates,
                            gallery_state,
                        ],
                        outputs=[results_store, similar_images_gallery],
                    )

            crop_model = None

            def load_model():
                nonlocal crop_model
                if crop_model is None:
                    crop_model = fasterrcnn_resnet50_fpn(
                        pretrained=True, cache_dir=cache_dir
                    )
                    crop_model.eval()
                return crop_model

            def crop_resize_with_scene(image, target_width, target_height):
                if image is None:
                    return None

                try:
                    model = load_model()
                    transform = transforms.Compose([transforms.ToTensor()])

                    with torch.no_grad():
                        detections = model(transform(image).unsqueeze(0))[0]

                    boxes = detections["boxes"]
                    scores = detections["scores"]
                    valid_boxes = boxes[scores > 0.5]

                    orig_width, orig_height = image.size

                    if len(valid_boxes) > 0:
                        x_min = int(torch.min(valid_boxes[:, 0]).item())
                        y_min = int(torch.min(valid_boxes[:, 1]).item())
                        x_max = int(torch.max(valid_boxes[:, 2]).item())
                        y_max = int(torch.max(valid_boxes[:, 3]).item())
                        center_x = (x_min + x_max) // 2
                        center_y = (y_min + y_max) // 2
                    else:
                        center_x = orig_width // 2
                        center_y = orig_height // 2

                    half_width = target_width // 2
                    half_height = target_height // 2

                    left = center_x - half_width
                    top = center_y - half_height
                    right = center_x + half_width
                    bottom = center_y + half_height

                    if left < 0:
                        center_x += abs(left)
                    elif right > orig_width:
                        center_x -= right - orig_width

                    if top < 0:
                        center_y += abs(top)
                    elif bottom > orig_height:
                        center_y -= bottom - orig_height

                    left = max(0, min(orig_width - target_width, center_x - half_width))
                    top = max(
                        0, min(orig_height - target_height, center_y - half_height)
                    )
                    right = left + target_width
                    bottom = top + target_height

                    return image.crop((left, top, right, bottom))
                except Exception as e:
                    k_log(f"Error in crop_resize_with_scene: {str(e)}")
                    return None

            def process_image_or_folder(image, folder_path, width, height):
                processed_gallery = []

                if folder_path:
                    folder_path = folder_path.strip()
                    if not os.path.exists(folder_path):
                        print(f"Folder not found: {folder_path}")
                        yield None, []
                        return

                    cropped_dir = os.path.join(folder_path, "cropped")
                    os.makedirs(cropped_dir, exist_ok=True)

                    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
                    file_list = sorted(
                        f
                        for f in os.listdir(folder_path)
                        if os.path.splitext(f)[1].lower() in valid_exts
                    )

                    for filename in file_list:
                        full_path = os.path.join(folder_path, filename)
                        try:
                            img = Image.open(full_path)
                            cropped = crop_resize_with_scene(img, width, height)
                            if cropped is None:
                                continue

                            cropped_filename = f"cropped_{filename}"
                            cropped_path = os.path.join(cropped_dir, cropped_filename)
                            cropped.save(cropped_path)

                            processed_gallery.append((cropped, f"Cropped: {filename}"))
                            yield cropped, processed_gallery

                        except Exception as e:
                            k_log(f"Error processing {full_path}: {str(e)}")

                    return

                if image is None:
                    yield None, []
                    return

                cropped_img = crop_resize_with_scene(image, width, height)
                if cropped_img is None:
                    yield None, []
                    return

                processed_gallery = [(cropped_img, "Cropped single image")]

                yield cropped_img, processed_gallery

            with gr.Tab("Semantic crop"):
                with gr.Row():
                    crop_image_input = gr.Image(
                        type="pil",
                        label="Single Image (ignored if folder path is used)",
                    )

                    with gr.Column():
                        crop_width = gr.Slider(
                            100, 2000, value=1024, step=1, label="Target width"
                        )
                        crop_height = gr.Slider(
                            100, 2000, value=1024, step=1, label="Target height"
                        )
                        crop_btn = gr.Button("Process Image(s)")
                        folder_input_crop = gr.Text(
                            label="Folder path (optional)",
                            placeholder="If set, all images in this folder will be processed",
                        )

                    current_image = gr.Image(label="Currently Processed Image")

                with gr.Row():
                    cropped_gallery = gr.Gallery(
                        label="Cropped Results (accumulated)", columns=5, height="auto"
                    )

                kubin.ui_utils.click_and_disable(
                    crop_btn,
                    fn=process_image_or_folder,
                    inputs=[
                        crop_image_input,
                        folder_input_crop,
                        crop_width,
                        crop_height,
                    ],
                    outputs=[current_image, cropped_gallery],
                    js=[
                        f"args => kubin.UI.taskStarted('{title}')",
                        f"args => kubin.UI.taskFinished('{title}')",
                    ],
                ).then(
                    fn=None,
                    _js='() => (kubin.notify.success("Cropping finished"))',
                    inputs=[],
                    outputs=[],
                )

        return image_tools_block

    return {
        "title": title,
        "tab_ui": lambda ui_s, ts: image_tools_ui(ui_s, ts),
    }
