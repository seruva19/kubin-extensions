from PIL import Image
import os
import gradio as gr
from similarity_search import ImageSimilaritySearch
import pandas as pd
import shutil

title = "Image Tools"


def setup(kubin):
    cache_dir = kubin.params("general", "cache_dir")

    def image_tools_ui(ui_shared, ui_tabs):
        with gr.Tabs() as image_tools_block:
            with gr.Tab("Similarity search"):
                current_folder = {"path": ""}

                def on_image_select(image_name, results_df):
                    if results_df is None or results_df.empty or not image_name:
                        return None, None, None

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
                            print(f"Query image not found: {query_path}")
                            return None, None, None

                        query_image = Image.open(query_path)

                        return (query_image, query_path, similar_images)
                    except Exception as e:
                        print(f"Error in on_image_select: {str(e)}")
                        return None, None, None

                def process_folder(
                    folder_path: str, top_results: int, score_threshold: float
                ):
                    try:
                        if not os.path.exists(folder_path):
                            print(f"Folder not found: {folder_path}")
                            return [pd.DataFrame(), gr.update(choices=[], value=None)]

                        current_folder["path"] = folder_path

                        search_engine = ImageSimilaritySearch()
                        results_df = search_engine.find_similar_images_batch(
                            folder_path, top_results, score_threshold
                        )

                        if not isinstance(results_df, pd.DataFrame):
                            results_df = pd.DataFrame(results_df)

                        unique_images = sorted(
                            results_df["Query Image"].unique().tolist()
                        )

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

                def remove_similar_image(
                    selected_gallery_item, image_name, results_df, remove_permanently
                ):
                    if (
                        not selected_gallery_item
                        or results_df is None
                        or results_df.empty
                    ):
                        return results_df, []

                    selected_item = selected_gallery_item[0][1]
                    image_filename = selected_item.split("Path: ")[-1]

                    if not image_filename:
                        return results_df, []

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
                            print(f"Moved duplicate image to: {new_duplicate_path}")
                        except Exception as e:
                            print(f"Error moving image {full_path}: {str(e)}")
                    else:
                        try:
                            if os.path.exists(full_path):
                                os.remove(full_path)
                                print(f"Deleted image: {full_path}")
                        except Exception as e:
                            print(f"Error deleting image {full_path}: {str(e)}")

                    new_df = results_df[
                        (results_df["Query Image"] != image_name)
                        & (results_df["Similar Image"] != image_name)
                    ]

                    unique_images = sorted(new_df["Query Image"].unique().tolist())
                    image_choices = [(img, img) for img in unique_images]
                    return new_df, gr.update(choices=image_choices, value=None), []

                with gr.Blocks() as similarity_block:
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

                        process_btn = gr.Button("Process Images")

                    results_store = gr.State()
                    selected_image_path = gr.State()

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
                                    resolve_btn = gr.Button(
                                        "Mark as Resolved", variant="primary"
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

                    def update_display(image_name, results_df):
                        selected_img, path, similar_df = on_image_select(
                            image_name, results_df
                        )

                        gallery_images = []
                        if similar_df is not None and not similar_df.empty:
                            for _, data in similar_df.iterrows():
                                full_path = os.path.join(
                                    current_folder["path"], data["Similar Image"]
                                )
                                try:
                                    img = Image.open(full_path)
                                    gallery_images.append(
                                        (
                                            img,
                                            f"Score: {data['Similarity Score']}, Rank: {data['Rank']}, Path: {data['Similar Image']}",
                                        )
                                    )
                                except Exception as e:
                                    print(f"Error loading image {full_path}: {str(e)}")

                        return [selected_img, gallery_images, path]

                    result_length = gr.Number(-1, visible=False)
                    process_btn.click(
                        fn=process_folder,
                        inputs=[folder_input, top_images, score_threshold],
                        outputs=[
                            results_store,
                            image_selector,
                            similar_images,
                            folder_input,
                            result_length,
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
                            similar_images_gallery,
                            selected_image_path,
                        ],
                    )

                    resolve_btn.click(
                        fn=mark_as_resolved,
                        inputs=[image_selector, results_store],
                        outputs=[results_store, image_selector, similar_images_gallery],
                    )

                    remove_btn.click(
                        fn=remove_similar_image,
                        inputs=[
                            similar_images_gallery,
                            image_selector,
                            results_store,
                            remove_duplicates,
                        ],
                        outputs=[results_store, image_selector, similar_images_gallery],
                    )

        return image_tools_block

    return {
        "title": title,
        "tab_ui": lambda ui_s, ts: image_tools_ui(ui_s, ts),
    }
