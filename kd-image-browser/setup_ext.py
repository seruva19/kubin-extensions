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

        return [
            gr.update(value=[], visible=True),
            gr.update(visible=False),
            gr.update(choices=[], value=None),
            gr.update(value="0 / 0"),
            gr.update(value=None),
            "No data found",
            "",
            [],
        ]

    def generate_video_metadata(video_path):
        from datetime import datetime
        import json

        try:
            stats = os.stat(video_path)
            filename = os.path.basename(video_path)
            filesize = stats.st_size / (1024 * 1024)  # MB
            created = datetime.fromtimestamp(stats.st_ctime).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            modified = datetime.fromtimestamp(stats.st_mtime).strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            metadata_html = f"""
            <div style="padding: 10px; background: #f5f5f5; border-radius: 5px; margin-top: 10px;">
                <h4 style="margin-top: 0;">Video Information</h4>
                <p><b>Filename:</b> {filename}</p>
                <p><b>Size:</b> {filesize:.2f} MB</p>
                <p><b>Created:</b> {created}</p>
                <p><b>Modified:</b> {modified}</p>
            </div>
            """

            try:
                from mutagen.mp4 import MP4

                video = MP4(video_path)
                comment = video.get("\xa9cmt", [None])[0]

                if comment:
                    try:
                        metadata = json.loads(comment)

                        metadata_html += f"""
                        <div style="padding: 10px; background: #e8f4f8; border-radius: 5px; margin-top: 10px;">
                            <h4 style="margin-top: 0;">Generation Parameters</h4>
                        """

                        for key, value in metadata.items():
                            metadata_html += f"<p><b>{key}:</b> {value}</p>"

                        metadata_html += "</div>"
                    except json.JSONDecodeError:
                        metadata_html += f"""
                        <div style="padding: 10px; background: #e8f4f8; border-radius: 5px; margin-top: 10px;">
                            <h4 style="margin-top: 0;">Generation Parameters</h4>
                            <pre style="white-space: pre-wrap; font-family: monospace; font-size: 12px;">{comment}</pre>
                        </div>
                        """
            except:
                pass

            return metadata_html
        except:
            return ""

    def view_folder(folder, sort_by, order_by):
        from datetime import datetime

        is_video_folder = folder and "video" in folder.lower()

        if is_video_folder:
            media_files = [
                entry.path
                for entry in os.scandir(f"{image_root}/{folder}")
                if entry.is_file()
                and entry.name.endswith(("mp4", "avi", "mov", "webm"))
            ]
        else:
            media_files = [
                entry.path
                for entry in os.scandir(f"{image_root}/{folder}")
                if entry.is_file() and entry.name.endswith(("png", "jpg", "jpeg"))
            ]

        if sort_by == "date":
            media_files = sorted(
                media_files,
                key=lambda f: os.path.getctime(f),
                reverse=order_by == "descending",
            )
        elif sort_by == "name":
            media_files = sorted(
                media_files,
                key=lambda f: str(os.path.splitext(os.path.basename(f))[0]).lower(),
                reverse=order_by == "descending",
            )

        if is_video_folder:
            video_choices = [(os.path.basename(f), f) for f in media_files]
            selected_video = media_files[0] if media_files else None
            video_count = len(media_files)
            index_text = f"1 / {video_count}" if video_count > 0 else "0 / 0"

            initial_metadata = ""
            if selected_video:
                initial_metadata = generate_video_metadata(selected_video)

            return [
                gr.update(value=None, visible=False),
                gr.update(visible=True),
                gr.update(choices=video_choices, value=selected_video),
                gr.update(value=index_text),
                gr.update(value=selected_video),
                gr.update(value=""),
                gr.update(value=initial_metadata),
                media_files,
            ]
        else:
            return [
                gr.update(value=media_files, visible=True),
                gr.update(visible=False),
                gr.update(choices=[], value=None),
                gr.update(value="0 / 0"),
                gr.update(value=None),
                gr.update(value=""),
                gr.update(value=""),
                media_files,
            ]

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
                    visible=True,
                )

                with gr.Column(visible=False) as video_browser:
                    with gr.Row():
                        video_list = gr.Dropdown(
                            label="Select Video",
                            choices=[],
                            interactive=True,
                            type="value",
                            scale=3,
                        )
                        video_index_display = gr.Textbox(
                            label="",
                            value="",
                            interactive=False,
                            scale=1,
                            show_label=False,
                        )

                    with gr.Row():
                        prev_video_btn = gr.Button("◀ Previous", size="sm", scale=1)
                        next_video_btn = gr.Button("Next ▶", size="sm", scale=1)

                    with gr.Row():
                        video_player = gr.Video(
                            label="Video player",
                            autoplay=False,
                            scale=2,
                        )
                        video_metadata = gr.HTML(
                            label="Video information",
                            scale=1,
                        )

                current_files = gr.State([])
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

                def on_video_select(video_path, all_videos):
                    if video_path and all_videos:
                        try:
                            current_index = all_videos.index(video_path) + 1
                            total = len(all_videos)
                            index_text = f"{current_index} / {total}"
                        except:
                            index_text = "? / ?"

                        metadata_html = generate_video_metadata(video_path)
                        return (
                            gr.update(value=index_text),
                            gr.update(value=video_path),
                            gr.update(value=metadata_html),
                        )
                    return (
                        gr.update(value="0 / 0"),
                        gr.update(value=None),
                        gr.update(value=""),
                    )

                def navigate_video(current_video, all_videos, direction):
                    if not all_videos or not current_video:
                        return current_video

                    try:
                        current_index = all_videos.index(current_video)
                        if direction == "prev":
                            new_index = (current_index - 1) % len(all_videos)
                        else:  # next
                            new_index = (current_index + 1) % len(all_videos)
                        return all_videos[new_index]
                    except:
                        return current_video

                video_list.change(
                    fn=on_video_select,
                    inputs=[video_list, current_files],
                    outputs=[video_index_display, video_player, video_metadata],
                    show_progress=False,
                )

                prev_video_btn.click(
                    fn=lambda v, files: navigate_video(v, files, "prev"),
                    inputs=[video_list, current_files],
                    outputs=[video_list],
                    show_progress=False,
                )

                next_video_btn.click(
                    fn=lambda v, files: navigate_video(v, files, "next"),
                    inputs=[video_list, current_files],
                    outputs=[video_list],
                    show_progress=False,
                )

                image_folders.change(
                    fn=view_folder,
                    inputs=[image_folders, image_sort, image_order],
                    outputs=[
                        folder_contents,
                        video_browser,
                        video_list,
                        video_index_display,
                        video_player,
                        metadata_info,
                        video_metadata,
                        current_files,
                    ],
                )
                image_sort.change(
                    fn=view_folder,
                    inputs=[image_folders, image_sort, image_order],
                    outputs=[
                        folder_contents,
                        video_browser,
                        video_list,
                        video_index_display,
                        video_player,
                        metadata_info,
                        video_metadata,
                        current_files,
                    ],
                )
                image_order.change(
                    fn=view_folder,
                    inputs=[image_folders, image_sort, image_order],
                    outputs=[
                        folder_contents,
                        video_browser,
                        video_list,
                        video_index_display,
                        video_player,
                        metadata_info,
                        video_metadata,
                        current_files,
                    ],
                )

                refresh_btn.click(
                    fn=check_folders,
                    inputs=[image_folders],
                    outputs=[no_folders_message, image_sources, image_folders],  # type: ignore
                    queue=False,
                ).then(
                    fn=refresh,
                    inputs=[image_folders, image_sort, image_order],
                    outputs=[
                        folder_contents,
                        video_browser,
                        video_list,
                        video_index_display,
                        video_player,
                        metadata_info,
                        video_metadata,
                        current_files,
                    ],
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
