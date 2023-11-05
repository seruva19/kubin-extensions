import gradio as gr
import os
from metadata_extractor import metadata_to_html
from pathlib import Path


def setup(kubin):
    image_root = kubin.params("general", "output_dir")

    yaml_config = kubin.yaml_utils.YamlConfig(Path(__file__).parent.absolute())
    config = yaml_config.read()

    def get_folders():
        return (
            [entry.name for entry in os.scandir(image_root) if entry.is_dir()]
            if os.path.exists(image_root)
            else []
        )

    def check_folders(folder):
        existing_folders = get_folders()
        exist = len(existing_folders) > 0
        choice = None if not exist else folder
        return [
            gr.update(visible=not exist),
            gr.update(visible=exist),
            gr.update(
                visible=exist,
                value=choice,
                choices=[folder for folder in existing_folders],
            ),
        ]

    def refresh(folder, sort_by, order_by):
        if folder is not None:
            folder_path = f"{image_root}/{folder}"
            if os.path.exists(folder_path):
                return view_folder(folder, sort_by, order_by)

        return [[], "No data found"]

    def view_folder(folder, sort_by, order_by):
        image_files = [
            entry.path
            for entry in os.scandir(f"{image_root}/{folder}")
            if entry.is_file() and entry.name.endswith(("png"))
        ]
        if sort_by == "date":
            image_files = sorted(
                image_files,
                key=lambda f: os.path.getctime(f),
                reverse=order_by == "descending",
            )
        elif sort_by == "name":
            image_files = sorted(
                image_files,
                key=lambda f: str(os.path.splitext(os.path.basename(f))[0]).lower(),
                reverse=order_by == "descending",
            )

        return [image_files, gr.update(value="")]

    def folder_contents_gallery_select(index, gallery):
        index = int(index, 10)
        html = metadata_to_html(gallery[index]["name"])

        return gr.update(value=html)

    def image_browser_ui(ui_shared, ui_tabs):
        folders = get_folders()

        with gr.Row() as image_browser_block:
            with gr.Column(scale=3) as folder_block:
                no_folders_message = gr.HTML(
                    "No image folders found", visible=len(folders) == 0
                )
                with gr.Box(visible=len(folders) > 0) as image_sources:
                    image_folders = gr.Radio(
                        [folder for folder in folders], label="Folder", interactive=True
                    )
                    image_sort = gr.Radio(
                        ["date", "name"],
                        value="date",
                        label="Sort images by",
                        interactive=True,
                    )
                    image_order = gr.Radio(
                        ["ascending", "descending"],
                        value="descending",
                        label="Sort order",
                        interactive=True,
                    )
                    show_amount = gr.Dropdown(
                        ["10", "20", "50", "100", "all"],
                        value="20",
                        label="How much images to show",
                        interactive=True,
                        scale=0,
                    )

                refresh_btn = gr.Button("Refresh", variant="secondary")
                metadata_info = gr.HTML()

            with gr.Column(scale=4):
                folder_contents = gr.Gallery(
                    label="Images in folder",
                    columns=5,
                    preview=False,
                    elem_classes="kd-image-browser-output",
                )
                with gr.Row() as page_nav:
                    first_page_btn = gr.Button(
                        "First page", variant="secondary", size="sm", scale=0
                    )
                    prev_page_btn = gr.Button(
                        "Previous page", variant="secondary", size="sm", scale=0
                    )
                    current_page = gr.Dropdown(
                        choices=["1"],
                        value="1",
                        label="Current page",
                        scale=0,
                        interactive=True,
                        show_label=False,
                    )
                    next_page_btn = gr.Button(
                        "Next page", variant="secondary", size="sm", scale=0
                    )
                    last_page_btn = gr.Button(
                        "Last page", variant="secondary", size="sm", scale=0
                    )
                page_nav.elem_classes = ["image-browser-page-navigation"]

                sender_index = gr.Textbox("-1", visible=False)

                ui_shared.create_base_send_targets(
                    folder_contents, "kd-image-browser-output", ui_tabs
                )
                ui_shared.create_ext_send_targets(
                    folder_contents, "kd-image-browser-output", ui_tabs
                )

                folder_contents.select(
                    fn=folder_contents_gallery_select,
                    _js=f"(si, fc) => ([kubin.UI.setImageIndex('kd-image-browser-output'), fc])",
                    inputs=[sender_index, folder_contents],
                    outputs=[metadata_info],
                    show_progress=False,
                )

                image_folders.change(
                    fn=view_folder,
                    inputs=[image_folders, image_sort, image_order],
                    outputs=[folder_contents, metadata_info],
                )
                image_sort.change(
                    fn=view_folder,
                    inputs=[image_folders, image_sort, image_order],
                    outputs=[folder_contents, metadata_info],
                )
                image_order.change(
                    fn=view_folder,
                    inputs=[image_folders, image_sort, image_order],
                    outputs=[folder_contents, metadata_info],
                )

                refresh_btn.click(
                    fn=check_folders,
                    inputs=[image_folders],
                    outputs=[no_folders_message, image_sources, image_folders],  # type: ignore
                    queue=False,
                ).then(
                    fn=refresh,
                    inputs=[image_folders, image_sort, image_order],
                    outputs=[folder_contents, metadata_info],
                )

        image_browser_block.elem_classes = [
            "kd-image-browser",
            "combined" if config["show_combined_preview"] else "",
        ]
        folder_block.elem_classes = ["block-params"]

        return image_browser_block

    def settings_ui():
        def save_changes(combined_preview):
            config["show_combined_preview"] = combined_preview
            yaml_config.write(config)

        with gr.Column() as settings_block:
            combined_preview = gr.Checkbox(
                lambda: config["show_combined_preview"],
                label="Show combined image preview",
                scale=0,
            )

            save_btn = gr.Button("Save settings", size="sm", scale=0)
            save_btn.click(
                save_changes, inputs=[combined_preview], outputs=[], queue=False
            ).then(fn=None, _js=("(x) => kubin.notify.success('Settings saved')"))

        settings_block.elem_classes = ["k-form"]
        return settings_block

    return {
        "title": "Image Browser",
        "tab_ui": lambda ui_s, ts: image_browser_ui(ui_s, ts),
        "settings_ui": settings_ui,
    }
