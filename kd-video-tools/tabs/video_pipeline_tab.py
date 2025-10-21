import gradio as gr
import json
import os
import pandas as pd
import re
from pathlib import Path
from functions.embedding_calculator import EmbeddingCalculator
from functions.quality_scorer import QualityScorer
from functions.video_processor import VideoProcessor


def parse_folder_list(folder_text):
    """
    Parse folder list from textbox.
    Format: "path/to/folder 50" or "path/to/folder"
    Returns list of tuples: [(path, number), ...]
    """
    folders = []
    for line in folder_text.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        # Try to match "folder number" pattern
        match = re.match(r"^(.+?)\s+(\d+)$", line)
        if match:
            folder_path, number = match.groups()
            folders.append((folder_path.strip(), int(number)))
        else:
            # Just a folder path without number
            folders.append((line, None))

    return folders


def pipeline_block(kubin, state, title):
    device = kubin.params("general", "device")

    embedding_calc = EmbeddingCalculator(kubin)
    quality_scorer = QualityScorer(kubin)
    video_processor = VideoProcessor(kubin)

    def calculate_quality_scores_tab(
        folder_list,
        quality_strategy,
        fastvqa_model,
        num_frames,
        batch_size,
        include_motion,
        include_subdirs,
        file_extensions,
        score_extension,
        overwrite_existing,
        min_dimension,
        progress=gr.Progress(),
    ):
        try:
            folders = parse_folder_list(folder_list)
            if not folders:
                return "Error: No valid folders specified", None

            progress(0.1, desc="Loading aesthetic model...")
            quality_scorer.load_aesthetic_model(device=device)

            progress(0.2, desc=f"Loading {fastvqa_model} model...")
            quality_scorer.load_fastvqa_model(model_type=fastvqa_model, device=device)

            progress(0.3, desc="Models loaded successfully")

            all_results = []
            total_assessed = 0
            total_skipped = 0
            total_failed = 0

            for idx, (folder_path, _) in enumerate(folders):
                progress(
                    (idx + 0.2) / len(folders),
                    desc=f"Processing {Path(folder_path).name}...",
                )

                result = quality_scorer.calculate_quality_scores_per_file(
                    video_dir=folder_path,
                    video_extensions=file_extensions,
                    num_frames=num_frames,
                    batch_size=batch_size,
                    include_subdirectories=include_subdirs,
                    quality_strategy=quality_strategy,
                    score_extension=score_extension,
                    overwrite_existing=overwrite_existing,
                    include_motion=include_motion,
                    min_dimension=min_dimension,
                )

                if "error" in result:
                    all_results.append(f"{folder_path}: {result['error']}")
                else:
                    total_assessed += result["assessed"]
                    total_skipped += result["skipped"]
                    total_failed += result["failed"]
                    all_results.append(
                        f"{folder_path}: {result['assessed']} assessed, "
                        f"{result['skipped']} skipped, {result['failed']} failed"
                    )

            progress(0.95, desc="Unloading models...")
            quality_scorer.unload_models()

            output_text = f"Quality scoring completed\n\n"
            output_text += f"Total folders processed: {len(folders)}\n"
            output_text += f"Total videos assessed: {total_assessed}\n"
            output_text += (
                f"Total videos skipped (already have .score): {total_skipped}\n"
            )
            output_text += f"Total videos failed: {total_failed}\n"
            output_text += f"Strategy used: {quality_strategy}\n\n"
            output_text += "Results per folder:\n"
            output_text += "\n".join(all_results)

            return output_text, None

        except Exception as e:
            return f"Error: {str(e)}", None

    def calculate_embeddings_tab(
        folder_list,
        model_name,
        num_frames,
        batch_size,
        include_subdirs,
        file_extensions,
        embedding_extension,
        overwrite_existing,
        progress=gr.Progress(),
    ):
        try:
            folders = parse_folder_list(folder_list)
            if not folders:
                return "Error: No valid folders specified", None

            progress(0, desc="Loading X-CLIP model...")

            # Load model
            embedding_calc.load_model(model_name=model_name, device=device)

            all_results = []
            total_processed = 0
            total_skipped = 0
            total_failed = 0

            for idx, (folder_path, _) in enumerate(folders):
                progress(
                    (idx + 0.2) / len(folders),
                    desc=f"Processing {Path(folder_path).name}...",
                )

                result = embedding_calc.calculate_embeddings_per_file(
                    video_dir=folder_path,
                    video_extensions=file_extensions,
                    num_frames=num_frames,
                    batch_size=batch_size,
                    include_subdirectories=include_subdirs,
                    embedding_extension=embedding_extension,
                    overwrite_existing=overwrite_existing,
                    model_name=model_name,
                )

                if "error" in result:
                    all_results.append(f"{folder_path}: {result['error']}")
                else:
                    total_processed += result["processed"]
                    total_skipped += result["skipped"]
                    total_failed += result["failed"]
                    all_results.append(
                        f"{folder_path}: {result['processed']} processed, "
                        f"{result['skipped']} skipped, {result['failed']} failed"
                    )

            embedding_calc.unload_model()

            output_text = f"Embedding calculation completed\n\n"
            output_text += f"Total folders processed: {len(folders)}\n"
            output_text += f"Total videos processed: {total_processed}\n"
            output_text += (
                f"Total videos skipped (already have .embedding): {total_skipped}\n"
            )
            output_text += f"Total videos failed: {total_failed}\n"
            output_text += f"Model used: {model_name}\n\n"
            output_text += "Results per folder:\n"
            output_text += "\n".join(all_results)

            return output_text, None

        except Exception as e:
            return f"Error: {str(e)}", None

    def estimate_selection_and_transform(
        folder_list,
        similarity_threshold,
        candidate_pool_multiplier,
        min_duration,
        max_duration,
        min_quality_score,
        score_extension,
        embedding_extension,
        progress=gr.Progress(),
    ):
        try:
            folders = parse_folder_list(folder_list)
            if not folders:
                return "Error: No valid folders specified", None

            all_results = []
            total_estimated = 0

            for idx, (folder_path, target_count) in enumerate(folders):
                folder_path = Path(folder_path)
                folder_name = folder_path.name

                progress(
                    (idx + 0.1) / len(folders), desc=f"Estimating {folder_name}..."
                )

                # Find all videos in folder
                all_videos = []
                for ext in [".mp4", ".avi", ".mov", ".mkv", ".webm"]:
                    all_videos.extend(folder_path.glob(f"*{ext}"))

                if not all_videos:
                    all_results.append(f"{folder_name}: No videos found → 0 selected")
                    continue

                # If no target count specified, use all videos in folder
                if target_count is None:
                    target_count = len(all_videos)

                # Load scores and embeddings from files (for filtering/selection)
                video_processor.load_quality_scores_from_files(
                    all_videos, score_extension
                )
                video_processor.load_embeddings_from_files(
                    all_videos, embedding_extension
                )

                # Apply filters (same logic as actual function)
                filtered_videos = all_videos.copy()

                # Filter by duration
                if min_duration > 0 or max_duration < 999:
                    duration_filtered = []
                    for video in filtered_videos:
                        duration = video_processor.get_video_duration(video)
                        if duration and min_duration <= duration <= max_duration:
                            duration_filtered.append(video)
                    filtered_videos = duration_filtered

                # Filter by minimum quality score
                if min_quality_score > 0:
                    quality_filtered = []
                    for video in filtered_videos:
                        if video in video_processor.quality_scores:
                            score = video_processor.quality_scores[video].get(
                                "final_score", 0
                            )
                            if score >= min_quality_score:
                                quality_filtered.append(video)
                    filtered_videos = quality_filtered

                # Select diverse high-quality videos (without copying)
                candidate_videos = [
                    video
                    for video in filtered_videos
                    if video in video_processor.embeddings
                    and video in video_processor.quality_scores
                ]

                if not candidate_videos:
                    all_results.append(
                        f"{folder_name}: No videos with required scores/embeddings → 0 selected"
                    )
                    continue

                missing_emb = len(
                    [v for v in filtered_videos if v not in video_processor.embeddings]
                )
                missing_scores = len(
                    [
                        v
                        for v in filtered_videos
                        if v not in video_processor.quality_scores
                    ]
                )
                warning_text = (
                    f" (missing emb: {missing_emb}, missing scores: {missing_scores})"
                    if (missing_emb or missing_scores)
                    else ""
                )

                selection_result = video_processor.select_diverse_videos(
                    similarity_threshold=similarity_threshold,
                    max_videos=target_count,
                    candidate_videos=candidate_videos,
                    candidate_pool_multiplier=candidate_pool_multiplier,
                )

                if "error" in selection_result:
                    all_results.append(
                        f"{folder_name}: {selection_result['error']} → 0 selected"
                    )
                    continue

                selected_count = selection_result["selected_count"]
                total_estimated += selected_count

                all_results.append(
                    f"{folder_name}: {selected_count} selected (from {len(all_videos)} total){warning_text}"
                )

            # Prepare output
            output_text = f"Selection estimation completed\n\n"
            output_text += f"Total folders processed: {len(folders)}\n"
            output_text += f"Total estimated selections: {total_estimated}\n\n"
            output_text += "Estimation per folder:\n"
            output_text += "\n".join(all_results)

            return output_text, None

        except Exception as e:
            return f"Error during estimation: {str(e)}", None

    def select_and_transform_tab(
        folder_list,
        similarity_threshold,
        candidate_pool_multiplier,
        min_duration,
        max_duration,
        min_quality_score,
        target_dir,
        resample_fps,
        preserve_structure,
        video_codec,
        preset,
        crf,
        audio_codec,
        score_extension,
        embedding_extension,
        progress=gr.Progress(),
    ):
        try:
            folders = parse_folder_list(folder_list)
            if not folders:
                return "Error: No valid folders specified", None

            all_results = []
            total_selected = 0

            # Determine base directory for structure preservation (calculate once)
            common_base_dir = None
            if preserve_structure:
                base_dirs = [Path(fp) for fp, _ in folders]
                if len(base_dirs) > 1:
                    # Find common parent directory
                    try:
                        common_base_dir = Path(
                            os.path.commonpath([str(d) for d in base_dirs])
                        )
                    except ValueError:
                        # If no common path, use the parent of the first directory
                        common_base_dir = base_dirs[0].parent
                else:
                    common_base_dir = base_dirs[0].parent

            for idx, (folder_path, target_count) in enumerate(folders):
                folder_path = Path(folder_path)
                folder_name = folder_path.name

                # Find all videos in folder
                all_videos = []
                for ext in [".mp4", ".avi", ".mov", ".mkv", ".webm"]:
                    all_videos.extend(folder_path.glob(f"*{ext}"))

                if not all_videos:
                    all_results.append(f"{folder_name}: No videos found")
                    continue

                # If no target count specified, use all videos in folder
                if target_count is None:
                    target_count = len(all_videos)  # Use total number of videos
                    all_results.append(
                        f"{folder_name}: Using all {target_count} videos (auto-detected)"
                    )

                progress(
                    (idx + 0.1) / len(folders), desc=f"Processing {folder_name}..."
                )

                # Load scores and embeddings from files
                video_processor.load_quality_scores_from_files(
                    all_videos, score_extension
                )
                video_processor.load_embeddings_from_files(
                    all_videos, embedding_extension
                )

                # Filter by duration
                if min_duration > 0 or max_duration < 999:
                    filtered_videos = []
                    for video in all_videos:
                        duration = video_processor.get_video_duration(video)
                        if duration and min_duration <= duration <= max_duration:
                            filtered_videos.append(video)
                    all_videos = filtered_videos

                # Filter by minimum quality score
                if min_quality_score > 0:
                    filtered_videos = []
                    for video in all_videos:
                        if video in video_processor.quality_scores:
                            score = video_processor.quality_scores[video].get(
                                "final_score", 0
                            )
                            if score >= min_quality_score:
                                filtered_videos.append(video)
                    all_videos = filtered_videos

                if not all_videos:
                    all_results.append(f"{folder_name}: No videos passed filters")
                    continue

                candidate_videos = [
                    video
                    for video in all_videos
                    if video in video_processor.embeddings
                    and video in video_processor.quality_scores
                ]

                if not candidate_videos:
                    all_results.append(
                        f"{folder_name}: No videos with both embeddings and quality scores"
                    )
                    continue

                missing_emb = len(
                    [v for v in all_videos if v not in video_processor.embeddings]
                )
                missing_scores = len(
                    [v for v in all_videos if v not in video_processor.quality_scores]
                )
                warning_msg = ""
                if missing_emb or missing_scores:
                    warning_msg = f" (missing embeddings: {missing_emb}, missing scores: {missing_scores})"

                # Select diverse high-quality videos
                selection_result = video_processor.select_diverse_videos(
                    similarity_threshold=similarity_threshold,
                    max_videos=target_count,
                    candidate_videos=candidate_videos,
                    candidate_pool_multiplier=candidate_pool_multiplier,
                )

                if "error" in selection_result:
                    all_results.append(f"{folder_name}: {selection_result['error']}")
                    continue

                selected_count = selection_result["selected_count"]
                total_selected += selected_count

                # Copy/resample if target directory specified
                if target_dir:
                    copy_result = video_processor.copy_selected_videos(
                        target_dir=target_dir,
                        resample_fps=resample_fps if resample_fps > 0 else None,
                        preserve_structure=preserve_structure,
                        base_dir=common_base_dir if preserve_structure else None,
                        video_codec=video_codec,
                        preset=preset,
                        crf=crf,
                        audio_codec=audio_codec,
                    )
                    all_results.append(
                        f"{folder_name}: Selected {selected_count}, "
                        f"copied {len(copy_result['success'])}, "
                        f"failed {len(copy_result['failed'])}{warning_msg}"
                    )
                else:
                    all_results.append(
                        f"{folder_name}: Selected {selected_count} videos{warning_msg}"
                    )

            # Prepare output
            output_text = f"Selection completed\n\n"
            output_text += f"Total folders processed: {len([f for f in folders if f[1] is not None])}\n"
            output_text += f"Total videos selected: {total_selected}\n"
            if target_dir:
                output_text += f"Target directory: {target_dir}\n"
            output_text += "\nResults per folder:\n"
            output_text += "\n".join(all_results)

            return output_text, None

        except Exception as e:
            return f"Error: {str(e)}", None

    with gr.Column() as pipeline_ui:
        folder_list_input = gr.TextArea(
            label="Folder List",
            placeholder="Enter one folder per line. Optional: add number for selection (Tab 3)\nExample:\nG:\\videos\\anime 50\nG:\\videos\\realistic 30",
            lines=5,
            value="",
        )

        with gr.Tabs():
            with gr.Tab("1. Quality Scoring"):
                gr.Markdown(
                    "Calculate quality scores for all videos and save as `.score` files"
                )

                with gr.Row():
                    qual_strategy = gr.Dropdown(
                        choices=[
                            "balanced",
                            "perceptual",
                            "technical",
                            "aesthetic",
                            "anime",
                            "adaptive",
                        ],
                        value="balanced",
                        label="Quality Strategy",
                    )

                    qual_fastvqa_model = gr.Dropdown(
                        choices=[
                            "FasterVQA",
                            "FasterVQA-MS",
                            "FasterVQA-MT",
                            "FAST-VQA",
                            "FAST-VQA-M",
                        ],
                        value="FasterVQA",
                        label="FAST-VQA Model",
                    )

                with gr.Row():
                    qual_num_frames = gr.Number(
                        value=8, minimum=1, maximum=32, label="Frames per Video"
                    )

                    qual_batch_size = gr.Slider(
                        minimum=1,
                        maximum=16,
                        step=1,
                        value=1,
                        label="FAST-VQA Batch Size",
                    )

                    qual_min_dimension = gr.Slider(
                        minimum=0,
                        maximum=1024,
                        step=32,
                        value=256,
                        label="Min Dimension (0=no resize)",
                    )

                    qual_include_motion = gr.Checkbox(
                        True, label="Include motion scoring"
                    )
                with gr.Row():
                    qual_include_subdirs = gr.Checkbox(
                        False, label="Include subdirectories"
                    )

                    qual_output_dir = gr.Textbox(value="", label="Output directory")
                    qual_file_extensions = gr.CheckboxGroup(
                        choices=[".mp4", ".avi", ".mov", ".mkv", ".webm"],
                        value=[".mp4"],
                        label="Video file extensions",
                    )
                with gr.Row():
                    qual_score_extension = gr.Textbox(
                        value=".score", label="Score file extension"
                    )

                    qual_overwrite = gr.Checkbox(
                        True, label="Overwrite existing .score files"
                    )

                    qual_calculate_btn = gr.Button(
                        "Calculate Quality Scores", variant="primary"
                    )

                qual_output = gr.Textbox(label="Results", lines=12, interactive=False)
                qual_dataframe = gr.Dataframe(
                    label="Status", interactive=False, visible=False
                )

            with gr.Tab("2. Embeddings"):
                gr.Markdown(
                    "Calculate embeddings for all videos and save as `.embedding` files"
                )

                with gr.Row():
                    emb_model_name = gr.Dropdown(
                        choices=[
                            "microsoft/xclip-base-patch32",
                            "microsoft/xclip-base-patch16",
                            "microsoft/xclip-large-patch14",
                        ],
                        value="microsoft/xclip-base-patch32",
                        label="X-CLIP Model",
                    )

                    emb_num_frames = gr.Number(
                        value=8, minimum=1, maximum=32, label="Frames per Video"
                    )

                    emb_batch_size = gr.Number(
                        value=4, minimum=1, maximum=16, label="Batch Size"
                    )
                with gr.Row():
                    emb_include_subdirs = gr.Checkbox(
                        False, label="Include subdirectories"
                    )

                    emb_file_extensions = gr.CheckboxGroup(
                        choices=[".mp4", ".avi", ".mov", ".mkv", ".webm"],
                        value=[".mp4"],
                        label="Video file extensions",
                    )

                    emb_embedding_extension = gr.Textbox(
                        value=".embedding", label="Embedding file extension"
                    )
                with gr.Row():
                    emb_overwrite = gr.Checkbox(
                        True, label="Overwrite existing .embedding files"
                    )

                    emb_calculate_btn = gr.Button(
                        "Calculate Embeddings", variant="primary"
                    )

                emb_output = gr.Textbox(label="Results", lines=12, interactive=False)
                emb_dataframe = gr.Dataframe(
                    label="Status", interactive=False, visible=False
                )

            with gr.Tab("3. Selection"):
                gr.Markdown(
                    "Select best diverse videos based on `.score` and `.embedding` files"
                )

                with gr.Row():
                    sel_similarity_threshold = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.80,
                        step=0.05,
                        label="Similarity Threshold (higher = more diverse)",
                    )

                    sel_candidate_multiplier = gr.Number(
                        value=3.0,
                        minimum=1.0,
                        maximum=10.0,
                        label="Candidate Pool Multiplier",
                    )

                    sel_min_duration = gr.Number(
                        value=1.0, minimum=0, label="Min Duration (seconds)"
                    )
                    sel_max_duration = gr.Number(
                        value=10.0, minimum=0, label="Max Duration (seconds)"
                    )

                with gr.Row():
                    sel_min_quality = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.5,
                        step=0.05,
                        label="Minimum Quality Score (0 = no filter)",
                    )

                    sel_target_dir = gr.Textbox(
                        label="Target Directory (leave empty to skip copying)",
                        placeholder="Path where selected videos will be copied",
                    )

                    sel_resample_fps = gr.Number(
                        value=16,
                        minimum=0,
                        maximum=120,
                        label="Resample to FPS (0 = no resampling)",
                    )

                    sel_preserve_structure = gr.Checkbox(
                        True, label="Preserve directory structure"
                    )

                with gr.Accordion("🎬 FFmpeg Settings", open=False):
                    with gr.Row():
                        sel_video_codec = gr.Dropdown(
                            choices=["libx264", "libx265", "copy", "mpeg4"],
                            value="libx264",
                            label="Video Codec",
                        )

                        sel_preset = gr.Dropdown(
                            choices=[
                                "ultrafast",
                                "superfast",
                                "veryfast",
                                "faster",
                                "fast",
                                "medium",
                                "slow",
                                "slower",
                                "veryslow",
                            ],
                            value="fast",
                            label="Encoding Preset",
                        )

                    with gr.Row():
                        sel_crf = gr.Slider(
                            minimum=0,
                            maximum=51,
                            value=23,
                            step=1,
                            label="Quality (CRF, lower=better)",
                        )

                        sel_audio_codec = gr.Dropdown(
                            choices=["copy", "aac", "mp3", "none"],
                            value="copy",
                            label="Audio Codec",
                        )
                with gr.Row():
                    sel_score_extension = gr.Textbox(
                        value=".score", label="Score file extension"
                    )

                    sel_embedding_extension = gr.Textbox(
                        value=".embedding", label="Embedding file extension"
                    )

                with gr.Row():
                    sel_estimate_btn = gr.Button(
                        "📊 Estimate Selection", variant="secondary"
                    )
                    sel_process_btn = gr.Button("Select Videos", variant="primary")

                sel_output = gr.Textbox(label="Results", lines=12, interactive=False)
                sel_dataframe = gr.Dataframe(
                    label="Selected Videos", interactive=False, visible=False
                )

        # Connect buttons
        kubin.ui_utils.click_and_disable(
            qual_calculate_btn,
            fn=calculate_quality_scores_tab,
            inputs=[
                folder_list_input,
                qual_strategy,
                qual_fastvqa_model,
                qual_num_frames,
                qual_batch_size,
                qual_include_motion,
                qual_include_subdirs,
                qual_file_extensions,
                qual_score_extension,
                qual_overwrite,
                qual_min_dimension,
            ],
            outputs=[qual_output, qual_dataframe],
            js=[
                f"args => kubin.UI.taskStarted('{title}')",
                f"args => kubin.UI.taskFinished('{title}')",
            ],
        )

        kubin.ui_utils.click_and_disable(
            emb_calculate_btn,
            fn=calculate_embeddings_tab,
            inputs=[
                folder_list_input,
                emb_model_name,
                emb_num_frames,
                emb_batch_size,
                emb_include_subdirs,
                emb_file_extensions,
                emb_embedding_extension,
                emb_overwrite,
            ],
            outputs=[emb_output, emb_dataframe],
            js=[
                f"args => kubin.UI.taskStarted('{title}')",
                f"args => kubin.UI.taskFinished('{title}')",
            ],
        )

        kubin.ui_utils.click_and_disable(
            sel_estimate_btn,
            fn=estimate_selection_and_transform,
            inputs=[
                folder_list_input,
                sel_similarity_threshold,
                sel_candidate_multiplier,
                sel_min_duration,
                sel_max_duration,
                sel_min_quality,
                sel_score_extension,
                sel_embedding_extension,
            ],
            outputs=[sel_output, sel_dataframe],
            js=[
                f"args => kubin.UI.taskStarted('{title}')",
                f"args => kubin.UI.taskFinished('{title}')",
            ],
        )

        kubin.ui_utils.click_and_disable(
            sel_process_btn,
            fn=select_and_transform_tab,
            inputs=[
                folder_list_input,
                sel_similarity_threshold,
                sel_candidate_multiplier,
                sel_min_duration,
                sel_max_duration,
                sel_min_quality,
                sel_target_dir,
                sel_resample_fps,
                sel_preserve_structure,
                sel_video_codec,
                sel_preset,
                sel_crf,
                sel_audio_codec,
                sel_score_extension,
                sel_embedding_extension,
            ],
            outputs=[sel_output, sel_dataframe],
            js=[
                f"args => kubin.UI.taskStarted('{title}')",
                f"args => kubin.UI.taskFinished('{title}')",
            ],
        )

    pipeline_ui.elem_classes = ["block-params"]
    return pipeline_ui
