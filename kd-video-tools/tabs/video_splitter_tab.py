import os
import subprocess
from pathlib import Path
import gradio as gr
import numpy as np


def splitter_block(kubin, state, title):
    def get_video_files_and_output(
        input_path_str, output_folder, include_subdirs, file_extensions
    ):
        # Ensure input_path_str is a valid path type
        if not isinstance(input_path_str, (str, bytes, os.PathLike)):
            raise ValueError(
                f"Input path must be a string, got {type(input_path_str).__name__}: {input_path_str}"
            )

        # Convert to string if needed
        input_path_str = str(input_path_str)
        input_path = Path(input_path_str)

        if input_path.is_file():
            # Single file: create folder with filename_splitted
            video_files = [input_path]
            input_base = input_path.parent

            if output_folder:
                output_base = Path(output_folder)
            else:
                output_base = input_path.parent

            return video_files, output_base, input_base

        elif input_path.is_dir():
            # Folder mode
            input_base = input_path

            if output_folder:
                output_base = Path(output_folder)
            else:
                output_base = input_path / "output"

            # Find video files
            video_files = []
            if include_subdirs:
                for ext in file_extensions:
                    video_files.extend(input_path.rglob(f"*{ext}"))
            else:
                for ext in file_extensions:
                    video_files.extend(input_path.glob(f"*{ext}"))

            return video_files, output_base, input_base

        raise ValueError(f"Input path does not exist: {input_path}")

    def split_with_pyscenedetect(
        input_folder,
        output_folder,
        detector_type,
        threshold,
        min_scene_len,
        show_progress,
        stats_file,
        save_images,
        image_format,
        num_images,
        image_name_template,
        output_format,
        filename_template,
        split_video,
        copy_codec,
        high_quality,
        rate_factor,
        preset,
        video_backend,
        downscale,
        frame_skip,
        weights,
        kernel_size,
        luma_only,
        adaptive_threshold,
        fade_bias,
        add_last_scene,
        merge_last_scene,
        include_subdirs,
        file_extensions,
        progress=gr.Progress(),
    ):
        try:
            if not input_folder or not os.path.exists(input_folder):
                return "Error: Input path does not exist or is not specified", None

            try:
                video_files, output_base, input_base = get_video_files_and_output(
                    input_folder, output_folder, include_subdirs, file_extensions
                )
            except ValueError as e:
                return f"Error: {str(e)}", None

            if not video_files:
                return f"Error: No video files found", None

            output_base.mkdir(parents=True, exist_ok=True)

            progress(0, desc=f"Found {len(video_files)} videos to process...")

            results = []
            total_scenes = 0
            failed_videos = []

            for idx, video_file in enumerate(video_files):
                video_name = video_file.stem

                progress(
                    (idx + 1) / len(video_files),
                    desc=f"Processing {video_name} ({idx + 1}/{len(video_files)})...",
                )

                video_output_dir = output_base / f"{video_name}_splitted"

                try:
                    video_output_dir.mkdir(parents=True, exist_ok=True)
                except Exception as mkdir_err:
                    failed_videos.append(video_name)
                    results.append(
                        f"{video_name}: Failed to create output directory - {str(mkdir_err)}"
                    )
                    continue

                cmd = ["scenedetect", "-i", str(video_file)]

                if video_backend:
                    cmd.extend(["--backend", video_backend])

                if downscale > 1:
                    cmd.extend(["--downscale", str(downscale)])

                if frame_skip > 0:
                    cmd.extend(["--frame-skip", str(frame_skip)])

                if stats_file:
                    stats_path = video_output_dir / f"{video_name}_stats.csv"
                    cmd.extend(["--stats", str(stats_path)])

                # Global scene length options
                if min_scene_len > 0:
                    cmd.extend(["--min-scene-len", str(int(min_scene_len))])

                if merge_last_scene:
                    cmd.append("--merge-last-scene")

                cmd.append("detect-" + detector_type)

                if detector_type == "content":
                    cmd.extend(["-t", str(threshold)])
                    if luma_only:
                        cmd.append("--luma-only")
                    if weights:
                        cmd.extend(["--weights", weights])
                    if kernel_size and int(kernel_size) > 0:
                        cmd.extend(["--kernel-size", str(int(kernel_size))])

                elif detector_type == "adaptive":
                    cmd.extend(["-t", str(adaptive_threshold)])
                    if luma_only:
                        cmd.append("--luma-only")
                    if weights:
                        cmd.extend(["--weights", weights])
                    if kernel_size and int(kernel_size) > 0:
                        cmd.extend(["--kernel-size", str(int(kernel_size))])

                elif detector_type == "threshold":
                    cmd.extend(["-t", str(threshold)])
                    if fade_bias is not None and fade_bias != 0:
                        cmd.extend(["--fade-bias", str(fade_bias)])
                    if add_last_scene:
                        cmd.append("--add-last-scene")

                cmd.append("list-scenes")
                if not show_progress:
                    cmd.append("--quiet")

                if save_images:
                    cmd.append("save-images")
                    cmd.extend(["-o", str(video_output_dir)])
                    if image_format:
                        cmd.extend(["-f", image_format])
                    if num_images > 0:
                        cmd.extend(["-n", str(num_images)])
                    if image_name_template:
                        cmd.extend(["--name-format", image_name_template])

                if split_video:
                    cmd.append("split-video")
                    cmd.extend(["-o", str(video_output_dir)])

                    if output_format:
                        cmd.extend(["-f", output_format])
                    if filename_template:
                        cmd.extend(["--filename", filename_template])

                    if copy_codec:
                        cmd.append("--copy")
                    elif high_quality:
                        cmd.append("--high-quality")
                    else:
                        if rate_factor:
                            cmd.extend(["-crf", str(rate_factor)])
                        if preset:
                            cmd.extend(["-p", preset])

                try:
                    # Use Popen to stream output in real-time while capturing it
                    process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                        universal_newlines=True,
                    )

                    output_lines = []
                    scene_count = 0

                    for line in process.stdout:
                        print(line, end="")
                        output_lines.append(line)

                        # Match "[PySceneDetect] Detected 94 scenes, average shot length..."
                        if "detected" in line.lower() and "scenes" in line.lower():
                            import re

                            match = re.search(
                                r"detected\s+(\d+)\s+scenes", line.lower()
                            )
                            if match:
                                scene_count = int(match.group(1))

                    return_code = process.wait()

                    if return_code == 0:
                        total_scenes += scene_count
                        results.append(f"{video_name}: {scene_count} scenes detected")
                    else:
                        failed_videos.append(video_name)
                        error_msg = (
                            "".join(output_lines[-10:])[:200]
                            if output_lines
                            else "Unknown error"
                        )
                        results.append(f"{video_name}: Failed - {error_msg}")

                except subprocess.TimeoutExpired:
                    failed_videos.append(video_name)
                    results.append(f"{video_name}: Timeout")
                except Exception as e:
                    failed_videos.append(video_name)
                    results.append(f"{video_name}: Error - {str(e)}")

            summary = f"PySceneDetect Processing Complete\n\n"
            summary += f"Total videos processed: {len(video_files)}\n"
            summary += f"Successful: {len(video_files) - len(failed_videos)}\n"
            summary += f"Failed: {len(failed_videos)}\n"
            summary += f"Total scenes detected: {total_scenes}\n"
            summary += f"Output location: {output_base}\n\n"
            summary += "Results:\n" + "\n".join(results)

            return summary, None

        except Exception as e:
            return f"Error: {str(e)}", None

    def detect_scenes_transnetv2(
        input_folder,
        output_folder,
        include_subdirs,
        file_extensions,
        threshold,
        visualize,
        progress=gr.Progress(),
    ):
        """Detect scenes using TransNetV2 and generate .predictions.txt and .scenes.txt files."""
        try:
            import sys

            parent_dir = os.path.dirname(os.path.dirname(__file__))
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)

            from transnet_v2.transnetv2 import (
                TransNetV2,
            )  #

            if not input_folder or not os.path.exists(input_folder):
                return "Error: Input path does not exist or is not specified", None

            try:
                video_files, output_base, input_base = get_video_files_and_output(
                    input_folder, output_folder, include_subdirs, file_extensions
                )
            except ValueError as e:
                return f"Error: {str(e)}", None

            if not video_files:
                return f"Error: No video files found", None

            # Create base output directory
            output_base.mkdir(parents=True, exist_ok=True)

            progress(0, desc="Loading TransNetV2 model...")

            try:
                model = TransNetV2()
            except Exception as e:
                return f"Error loading TransNetV2 model: {str(e)}", None

            progress(0.1, desc=f"Found {len(video_files)} videos to process...")

            results = []
            total_scenes = 0
            failed_videos = []

            for idx, video_file in enumerate(video_files):
                video_name = video_file.stem

                progress(
                    0.1 + (idx + 1) / len(video_files) * 0.9,
                    desc=f"Processing {video_name} ({idx + 1}/{len(video_files)})...",
                )

                # Create output directory: videoname_splitted
                video_output_dir = output_base / f"{video_name}_splitted"

                try:
                    video_output_dir.mkdir(parents=True, exist_ok=True)
                except Exception as mkdir_err:
                    failed_videos.append(video_name)
                    results.append(
                        f"{video_name}: Failed to create output directory - {str(mkdir_err)}"
                    )
                    continue

                predictions_file = video_output_dir / f"{video_name}.predictions.txt"
                scenes_file = video_output_dir / f"{video_name}.scenes.txt"
                vis_file = video_output_dir / f"{video_name}.vis.png"

                # Check if already processed
                if predictions_file.exists() and scenes_file.exists():
                    try:
                        scenes_data = np.loadtxt(scenes_file, dtype=np.int32)
                        scene_count = (
                            1 if len(scenes_data.shape) == 1 else scenes_data.shape[0]
                        )
                        total_scenes += scene_count
                        results.append(
                            f"⊙ {video_name}: Already processed ({scene_count} scenes)"
                        )
                        continue
                    except:
                        pass

                # Process video
                try:
                    video_frames, single_frame_predictions, all_frame_predictions = (
                        model.predict_video(str(video_file))
                    )

                    predictions = np.stack(
                        [single_frame_predictions, all_frame_predictions], 1
                    )
                    np.savetxt(predictions_file, predictions, fmt="%.6f")

                    scenes = model.predictions_to_scenes(
                        single_frame_predictions, threshold=threshold
                    )
                    np.savetxt(scenes_file, scenes, fmt="%d")

                    scene_count = scenes.shape[0]
                    total_scenes += scene_count

                    if visualize:
                        try:
                            pil_image = model.visualize_predictions(
                                video_frames,
                                predictions=(
                                    single_frame_predictions,
                                    all_frame_predictions,
                                ),
                            )
                            pil_image.save(vis_file)
                            results.append(
                                f"{video_name}: {scene_count} scenes detected (visualization saved)"
                            )
                        except Exception as vis_err:
                            results.append(
                                f"{video_name}: {scene_count} scenes detected (visualization failed: {str(vis_err)})"
                            )
                    else:
                        results.append(f"{video_name}: {scene_count} scenes detected")

                except Exception as e:
                    failed_videos.append(video_name)
                    error_msg = str(e)[:200]
                    results.append(f"{video_name}: Error - {error_msg}")

            summary = f"TransNetV2 Scene Detection Complete\n\n"
            summary += f"Total videos processed: {len(video_files)}\n"
            summary += f"Successful: {len(video_files) - len(failed_videos)}\n"
            summary += f"Failed: {len(failed_videos)}\n"
            summary += f"Total scenes detected: {total_scenes}\n"
            summary += f"Output location: {output_base}\n\n"
            summary += (
                f"Generated files: *.scenes.txt, *.predictions.txt"
                + (" *.vis.png" if visualize else "")
                + "\n\n"
            )
            summary += "Results:\n" + "\n".join(results)

            # Add scene details to text output
            summary += "\n\n" + "=" * 80 + "\n"
            summary += "SCENE DETAILS\n"
            summary += "=" * 80 + "\n"

            for video_file in video_files:
                video_name = video_file.stem
                video_output_dir = output_base / f"{video_name}_splitted"
                scenes_file = video_output_dir / f"{video_name}.scenes.txt"

                if scenes_file.exists():
                    summary += f"\nVideo: {video_name}\n"
                    summary += f"Scenes file: {scenes_file}\n\n"
                    try:
                        with open(scenes_file, "r") as f:
                            scene_lines = f.readlines()
                        summary += f"Total scenes: {len(scene_lines)}\n"
                        summary += (
                            f"{'Scene':<8} {'Start':<10} {'End':<10} {'Duration':<10}\n"
                        )
                        summary += "-" * 40 + "\n"
                        for idx, line in enumerate(scene_lines):
                            line = line.strip()
                            if line:
                                parts = line.split()
                                if len(parts) == 2:
                                    start_frame = int(parts[0])
                                    end_frame = int(parts[1])
                                    duration = end_frame - start_frame + 1
                                    summary += f"{idx+1:<8} {start_frame:<10} {end_frame:<10} {duration:<10}\n"
                        summary += "\n"
                    except Exception as e:
                        summary += f"Error reading scenes file: {str(e)}\n\n"

            return summary, None

        except Exception as e:
            return f"Error: {str(e)}", None

    def split_by_scenes(
        input_folder,
        output_folder,
        include_subdirs,
        file_extensions,
        video_codec,
        preset,
        crf,
        audio_codec,
        copy_codec,
        progress=gr.Progress(),
    ):
        """Split videos using existing .scenes.txt files generated by TransNetV2."""
        try:
            import ffmpeg
            import numpy as np

            if not input_folder or not os.path.exists(input_folder):
                return "Error: Input path does not exist or is not specified", None

            try:
                video_files, output_base, input_base = get_video_files_and_output(
                    input_folder, output_folder, include_subdirs, file_extensions
                )
            except ValueError as e:
                return f"Error: {str(e)}", None

            if not video_files:
                return f"Error: No video files found", None

            # Create base output directory
            output_base.mkdir(parents=True, exist_ok=True)

            progress(0, desc=f"Found {len(video_files)} videos to split...")

            results = []
            total_scenes_split = 0
            failed_videos = []
            skipped_videos = []

            for idx, video_file in enumerate(video_files):
                video_name = video_file.stem

                progress(
                    (idx + 1) / len(video_files),
                    desc=f"Splitting {video_name} ({idx + 1}/{len(video_files)})...",
                )

                # Look for scenes file in: videoname_splitted folder
                video_output_dir = output_base / f"{video_name}_splitted"
                scenes_file = video_output_dir / f"{video_name}.scenes.txt"

                if not scenes_file.exists():
                    skipped_videos.append(video_name)
                    results.append(
                        f"⊘ {video_name}: No .scenes.txt file found - run detection first"
                    )
                    continue

                # Load scenes
                try:
                    scenes_data = np.loadtxt(scenes_file, dtype=np.int32, ndmin=2)
                except Exception as e:
                    failed_videos.append(video_name)
                    results.append(
                        f"{video_name}: Failed to load scenes file - {str(e)}"
                    )
                    continue

                # Get video FPS
                try:
                    probe = ffmpeg.probe(str(video_file))
                    video_stream = next(
                        (s for s in probe["streams"] if s["codec_type"] == "video"),
                        None,
                    )
                    if not video_stream:
                        failed_videos.append(video_name)
                        results.append(f"{video_name}: No video stream found")
                        continue

                    fps_str = video_stream.get("r_frame_rate", "30/1")
                    fps_parts = fps_str.split("/")
                    fps = float(fps_parts[0]) / float(fps_parts[1])

                except Exception as e:
                    failed_videos.append(video_name)
                    results.append(f"{video_name}: Failed to probe video - {str(e)}")
                    continue

                # Ensure output directory exists
                try:
                    video_output_dir.mkdir(parents=True, exist_ok=True)
                except Exception as mkdir_err:
                    failed_videos.append(video_name)
                    results.append(
                        f"{video_name}: Failed to create output directory - {str(mkdir_err)}"
                    )
                    continue

                # Split video by scenes
                scene_count = 0
                failed_scenes = 0

                for scene_idx, (start_frame, end_frame) in enumerate(scenes_data):
                    scene_num = scene_idx + 1
                    output_file = (
                        video_output_dir / f"{video_name}_scene_{scene_num:04d}.mp4"
                    )

                    start_time = start_frame / fps
                    duration = (end_frame - start_frame + 1) / fps

                    try:
                        input_stream = ffmpeg.input(
                            str(video_file), ss=start_time, t=duration
                        )

                        if copy_codec:
                            output_stream = ffmpeg.output(
                                input_stream,
                                str(output_file),
                                codec="copy",
                                avoid_negative_ts="make_zero",
                            )
                        else:
                            output_args = {
                                "vcodec": video_codec,
                                "crf": crf,
                                "preset": preset,
                            }

                            if audio_codec == "none":
                                output_args["an"] = None
                            elif audio_codec == "copy":
                                output_args["acodec"] = "copy"
                            else:
                                output_args["acodec"] = audio_codec

                            output_stream = ffmpeg.output(
                                input_stream, str(output_file), **output_args
                            )

                        ffmpeg.run(
                            output_stream,
                            overwrite_output=True,
                            quiet=False,
                        )
                        scene_count += 1

                    except ffmpeg.Error as e:
                        failed_scenes += 1
                        print(
                            f"[ERROR] Failed to split scene {scene_num}: {str(e.stderr.decode() if e.stderr else e)}"
                        )
                    except Exception as e:
                        failed_scenes += 1
                        print(f"[ERROR] Failed to split scene {scene_num}: {str(e)}")

                total_scenes_split += scene_count

                if scene_count > 0:
                    if failed_scenes > 0:
                        results.append(
                            f"{video_name}: {scene_count} scenes split, {failed_scenes} failed"
                        )
                    else:
                        results.append(
                            f"{video_name}: {scene_count} scenes split successfully"
                        )
                else:
                    failed_videos.append(video_name)
                    results.append(f"{video_name}: All scenes failed to split")

            summary = f"Video Splitting by Scenes Complete\n\n"
            summary += f"Total videos processed: {len(video_files)}\n"
            summary += f"Successful: {len(video_files) - len(failed_videos) - len(skipped_videos)}\n"
            summary += f"Skipped (no scenes file): {len(skipped_videos)}\n"
            summary += f"Failed: {len(failed_videos)}\n"
            summary += f"Total scenes split: {total_scenes_split}\n"
            summary += f"Output location: {output_base}\n\n"
            summary += "Results:\n" + "\n".join(results)

            # Add split scene details to text output
            summary += "\n\n" + "=" * 80 + "\n"
            summary += "SPLIT SCENE DETAILS\n"
            summary += "=" * 80 + "\n"

            for video_file in video_files:
                video_name = video_file.stem
                video_output_dir = output_base / f"{video_name}_splitted"
                scenes_file = video_output_dir / f"{video_name}.scenes.txt"

                if scenes_file.exists():
                    summary += f"\nVideo: {video_name}\n"
                    summary += f"Output directory: {video_output_dir}\n\n"
                    try:
                        # Get video FPS for time calculation
                        try:
                            probe = ffmpeg.probe(str(video_file))
                            video_stream = next(
                                (
                                    s
                                    for s in probe["streams"]
                                    if s["codec_type"] == "video"
                                ),
                                None,
                            )
                            if video_stream:
                                fps_str = video_stream.get("r_frame_rate", "30/1")
                                fps_parts = fps_str.split("/")
                                fps = float(fps_parts[0]) / float(fps_parts[1])
                            else:
                                fps = 30.0
                        except:
                            fps = 30.0

                        with open(scenes_file, "r") as f:
                            scene_lines = f.readlines()

                        summary += f"Total scenes: {len(scene_lines)}\n"
                        summary += f"Video FPS: {fps:.2f}\n\n"
                        summary += f"{'Scene':<8} {'Start':<10} {'End':<10} {'Frames':<10} {'Duration':<12} {'Output File':<40} {'Status':<6}\n"
                        summary += "-" * 100 + "\n"

                        for idx, line in enumerate(scene_lines):
                            line = line.strip()
                            if line:
                                parts = line.split()
                                if len(parts) == 2:
                                    start_frame = int(parts[0])
                                    end_frame = int(parts[1])
                                    duration_frames = end_frame - start_frame + 1
                                    duration_secs = duration_frames / fps
                                    scene_num = idx + 1
                                    output_file = (
                                        video_output_dir
                                        / f"{video_name}_scene_{scene_num:04d}.mp4"
                                    )
                                    status = "+" if output_file.exists() else ""

                                    summary += f"{scene_num:<8} {start_frame:<10} {end_frame:<10} {duration_frames:<10} {duration_secs:<12.2f} {output_file.name:<40} {status:<6}\n"
                        summary += "\n"
                    except Exception as e:
                        summary += f"Error processing scenes: {str(e)}\n\n"

            return summary, None

        except Exception as e:
            return f"Error: {str(e)}", None

    with gr.Column() as splitter_ui:
        with gr.Row():
            input_folder = gr.Textbox(
                label="Input Path (File or Folder)",
                placeholder="Path to video file OR folder containing videos",
                scale=3,
            )
            output_folder = gr.Textbox(
                label="Output Folder (optional)",
                placeholder="Leave empty: file mode → same location, folder mode → creates 'output' subfolder",
                scale=3,
            )

        with gr.Row():
            file_extensions = gr.CheckboxGroup(
                choices=[".mp4", ".avi", ".mov", ".mkv", ".webm"],
                value=[".mp4"],
                label="Video file extensions",
            )
            include_subdirs = gr.Checkbox(
                False,
                label="Include subdirectories",
            )

        with gr.Tabs() as method_tabs:
            with gr.Tab("PySceneDetect Settings", id="pyscenedetect_tab"):
                detector_type = gr.Dropdown(
                    choices=["content", "adaptive", "threshold"],
                    value="content",
                    label="Detector Type",
                    info="content: Content-aware detection | adaptive: Adaptive threshold | threshold: Simple threshold",
                )

                with gr.Accordion("🎯 Detection Parameters", open=True):
                    with gr.Row():
                        threshold = gr.Slider(
                            minimum=0,
                            maximum=255,
                            value=27,
                            step=1,
                            label="Threshold",
                            info="Detection sensitivity (lower = more sensitive)",
                        )
                        min_scene_len = gr.Number(
                            value=15,
                            minimum=1,
                            label="Min Scene Length (frames)",
                            info="Minimum number of frames per scene",
                        )

                    with gr.Row():
                        weights = gr.Textbox(
                            label="Color Weights",
                            placeholder="e.g., 1.0 1.0 1.0 (R G B) or leave empty for auto",
                            info="RGB channel weights for content detection",
                        )
                        adaptive_threshold = gr.Slider(
                            minimum=0,
                            maximum=10,
                            value=3.0,
                            step=0.1,
                            label="Adaptive Threshold",
                            info="Threshold multiplier (adaptive detector only)",
                        )
                        kernel_size = gr.Number(
                            value=-1,
                            label="Kernel Size",
                            info="Kernel size for image processing (-1 for auto)",
                        )

                    with gr.Row():
                        luma_only = gr.Checkbox(
                            False,
                            label="Luma Only",
                            info="Use only luma (brightness) channel",
                        )
                        fade_bias = gr.Slider(
                            minimum=-1,
                            maximum=1,
                            value=0,
                            step=0.1,
                            label="Fade Bias",
                            info="Bias for fade detection (threshold detector)",
                        )

                    with gr.Row():
                        add_last_scene = gr.Checkbox(
                            True,
                            label="Add Last Scene",
                            info="Include final scene (threshold detector)",
                        )
                        merge_last_scene = gr.Checkbox(
                            False,
                            label="Merge Last Scene",
                            info="Merge short last scene (threshold detector)",
                        )

                with gr.Accordion("⚙️ Backend & Performance", open=True):
                    with gr.Row():
                        video_backend = gr.Dropdown(
                            choices=["opencv", "pyav", ""],
                            value="opencv",
                            label="Video Backend",
                            info="Backend for video processing",
                        )
                        downscale = gr.Number(
                            value=1,
                            minimum=1,
                            maximum=8,
                            label="Downscale Factor",
                            info="Downscale video for faster processing (1 = no downscale)",
                        )
                        frame_skip = gr.Number(
                            value=0,
                            minimum=0,
                            label="Frame Skip",
                            info="Skip frames for faster processing (0 = no skip)",
                        )

                with gr.Accordion("📊 Output Options", open=True):
                    with gr.Row():
                        show_progress = gr.Checkbox(
                            True,
                            label="Show Progress",
                            info="Display progress during processing",
                        )
                        stats_file = gr.Checkbox(
                            True,
                            label="Save Stats File",
                            info="Save statistics CSV file",
                        )

                    with gr.Row():
                        save_images = gr.Checkbox(
                            False,
                            label="Save Scene Images",
                            info="Extract images from detected scenes",
                        )
                        image_format = gr.Dropdown(
                            choices=["jpeg", "png", "webp"],
                            value="jpeg",
                            label="Image Format",
                        )
                        num_images = gr.Number(
                            value=3,
                            minimum=0,
                            label="Images per Scene",
                            info="Number of images to save per scene (0 = all)",
                        )

                    image_name_template = gr.Textbox(
                        value="$VIDEO_NAME-Scene-$SCENE_NUMBER-$IMAGE_NUMBER",
                        label="Image Name Template",
                        info="Template for image filenames",
                    )

                    with gr.Row():
                        split_video = gr.Checkbox(
                            True,
                            label="Split Video Files",
                            info="Split videos into separate scene files",
                        )
                        output_format = gr.Dropdown(
                            choices=["mp4", "mkv", "avi", "webm", ""],
                            value="mp4",
                            label="Output Format",
                        )

                    filename_template = gr.Textbox(
                        value="$VIDEO_NAME-Scene-$SCENE_NUMBER",
                        label="Filename Template",
                        info="Template for split video filenames",
                    )

                with gr.Accordion("🎬 Encoding Settings", open=True):
                    with gr.Row():
                        copy_codec = gr.Checkbox(
                            False,
                            label="Copy Codec",
                            info="Copy without re-encoding (fastest)",
                        )
                        high_quality = gr.Checkbox(
                            False,
                            label="High Quality",
                            info="Use high quality preset",
                        )

                    with gr.Row():
                        rate_factor = gr.Slider(
                            minimum=0,
                            maximum=51,
                            value=22,
                            step=1,
                            label="CRF (Rate Factor)",
                            info="Quality level (lower = better quality, larger file)",
                        )
                        preset = gr.Dropdown(
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
                            value="medium",
                            label="Encoding Preset",
                            info="Speed vs compression trade-off",
                        )

                pyscenedetect_btn = gr.Button(
                    "Split with PySceneDetect", variant="primary", size="lg"
                )

            with gr.Tab("TransNetV2 Settings", id="transnetv2_tab"):
                with gr.Accordion("🎯 Detection Settings", open=True):
                    with gr.Row():
                        tn_threshold = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.5,
                            step=0.05,
                            label="Detection Threshold",
                            info="Scene boundary sensitivity (lower = more scenes)",
                        )
                        tn_visualize = gr.Checkbox(
                            False,
                            label="Generate Visualization",
                            info="Create .vis.png files showing predictions",
                        )

                detect_btn = gr.Button(
                    "🔍 Detect Scenes (Generate .scenes.txt files)",
                    variant="primary",
                    size="lg",
                )

                with gr.Accordion("🎬 Splitting Settings", open=True):
                    with gr.Row():
                        tn_copy_codec = gr.Checkbox(
                            True,
                            label="Copy Codec (Fast)",
                            info="Copy without re-encoding (recommended)",
                        )
                        tn_video_codec = gr.Dropdown(
                            choices=["libx264", "libx265", "copy", "mpeg4"],
                            value="libx264",
                            label="Video Codec",
                            info="Used only if Copy Codec is disabled",
                        )

                    with gr.Row():
                        tn_preset = gr.Dropdown(
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
                            info="Speed vs compression trade-off",
                        )
                        tn_crf = gr.Slider(
                            minimum=0,
                            maximum=51,
                            value=23,
                            step=1,
                            label="Quality (CRF)",
                            info="Lower = better quality (18-28 recommended)",
                        )

                    with gr.Row():
                        tn_audio_codec = gr.Dropdown(
                            choices=["copy", "aac", "mp3", "none"],
                            value="copy",
                            label="Audio Codec",
                            info="How to handle audio tracks",
                        )

                split_btn = gr.Button(
                    "✂️ Split Videos by Detected Scenes",
                    variant="secondary",
                    size="lg",
                )

        # Output area
        with gr.Row():
            output_text = gr.Textbox(
                label="Processing Results",
                lines=15,
                interactive=False,
                show_copy_button=True,
            )

        output_dataframe = gr.Dataframe(
            label="Scene Details",
            interactive=False,
            visible=False,
        )

        kubin.ui_utils.click_and_disable(
            pyscenedetect_btn,
            fn=split_with_pyscenedetect,
            inputs=[
                input_folder,
                output_folder,
                detector_type,
                threshold,
                min_scene_len,
                show_progress,
                stats_file,
                save_images,
                image_format,
                num_images,
                image_name_template,
                output_format,
                filename_template,
                split_video,
                copy_codec,
                high_quality,
                rate_factor,
                preset,
                video_backend,
                downscale,
                frame_skip,
                weights,
                kernel_size,
                luma_only,
                adaptive_threshold,
                fade_bias,
                add_last_scene,
                merge_last_scene,
                include_subdirs,
                file_extensions,
            ],
            outputs=[output_text, output_dataframe],
            js=[
                f"args => kubin.UI.taskStarted('{title}')",
                f"args => kubin.UI.taskFinished('{title}')",
            ],
        )

        kubin.ui_utils.click_and_disable(
            detect_btn,
            fn=detect_scenes_transnetv2,
            inputs=[
                input_folder,
                output_folder,
                include_subdirs,
                file_extensions,
                tn_threshold,
                tn_visualize,
            ],
            outputs=[output_text, output_dataframe],
            js=[
                f"args => kubin.UI.taskStarted('{title}')",
                f"args => kubin.UI.taskFinished('{title}')",
            ],
        )

        kubin.ui_utils.click_and_disable(
            split_btn,
            fn=split_by_scenes,
            inputs=[
                input_folder,
                output_folder,
                include_subdirs,
                file_extensions,
                tn_video_codec,
                tn_preset,
                tn_crf,
                tn_audio_codec,
                tn_copy_codec,
            ],
            outputs=[output_text, output_dataframe],
            js=[
                f"args => kubin.UI.taskStarted('{title}')",
                f"args => kubin.UI.taskFinished('{title}')",
            ],
        )

    splitter_ui.elem_classes = ["block-params"]
    return splitter_ui
