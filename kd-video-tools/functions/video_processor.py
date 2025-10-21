import math
import subprocess
import shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm


class VideoProcessor:
    def __init__(self, kubin):
        self.kubin = kubin
        self.embeddings = {}
        self.quality_scores = {}
        self.selected_videos = []

    def load_embeddings(self, embeddings_dict):
        self.embeddings = {Path(k): v for k, v in embeddings_dict.items()}

    def load_quality_scores(self, quality_dict):
        self.quality_scores = {Path(k): v for k, v in quality_dict.items()}

    def load_embeddings_from_files(self, video_paths, embedding_extension=".embedding"):
        import json

        self.embeddings = {}
        loaded_count = 0
        missing_count = 0

        for video_path in video_paths:
            video_path = Path(video_path)
            embedding_file = video_path.parent / (video_path.name + embedding_extension)

            if embedding_file.exists():
                try:
                    with open(embedding_file, "r") as f:
                        data = json.load(f)
                        self.embeddings[video_path] = data.get("embedding")
                        loaded_count += 1
                except Exception as e:
                    missing_count += 1
            else:
                missing_count += 1
        return loaded_count

    def load_quality_scores_from_files(self, video_paths, score_extension=".score"):
        import json

        self.quality_scores = {}
        loaded_count = 0
        missing_count = 0

        for video_path in video_paths:
            video_path = Path(video_path)
            score_file = video_path.parent / (video_path.name + score_extension)

            if score_file.exists():
                try:
                    with open(score_file, "r") as f:
                        self.quality_scores[video_path] = json.load(f)
                        loaded_count += 1
                except Exception as e:
                    missing_count += 1
            else:
                missing_count += 1
        return loaded_count

    def cosine_similarity(self, a, b):
        a, b = np.array(a), np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def select_diverse_videos(
        self,
        similarity_threshold=0.80,
        max_videos=None,
        quality_weight=0.7,
        diversity_weight=0.3,
        candidate_videos=None,
        candidate_pool_multiplier=3.0,
    ):
        """
        Select diverse videos based on embeddings and quality scores.

        Args:
            similarity_threshold: Threshold for considering videos as similar
            max_videos: Maximum number of videos to select
            quality_weight: Weight for quality in selection (0-1)
            diversity_weight: Weight for diversity in selection (0-1)
        """
        if not self.embeddings:
            return {"error": "No embeddings loaded"}

        if not self.quality_scores:
            return {"error": "No quality scores loaded"}

        if candidate_videos is not None:
            candidate_set = {Path(v) for v in candidate_videos}
        else:
            candidate_set = set(self.embeddings.keys())

        # Get videos that have both embeddings and quality scores
        valid_videos = [
            v
            for v in candidate_set
            if v in self.embeddings and v in self.quality_scores
        ]

        if not valid_videos:
            return {"error": "No videos with both embeddings and quality scores"}

        # Sort by quality first
        video_qualities = [
            (v, self.quality_scores[v]["final_score"]) for v in valid_videos
        ]
        video_qualities.sort(key=lambda x: x[1], reverse=True)

        if (
            max_videos
            and candidate_pool_multiplier
            and candidate_pool_multiplier > 1
            and len(video_qualities) > max_videos
        ):
            pool_size = int(math.ceil(max_videos * candidate_pool_multiplier))
            pool_size = max(max_videos, pool_size)
            pool_size = min(pool_size, len(video_qualities))
            video_qualities = video_qualities[:pool_size]

        # Select diverse videos
        selected = []
        selected_embeddings = []

        for video, quality in tqdm(video_qualities, desc="Selecting diverse videos"):
            video_emb = self.embeddings[video]

            # Check similarity with already selected videos
            if len(selected_embeddings) > 0:
                similarities = [
                    self.cosine_similarity(video_emb, emb)
                    for emb in selected_embeddings
                ]
                max_sim = max(similarities)

                # Skip if too similar to any selected video
                if max_sim > similarity_threshold:
                    continue

            # Add to selection
            selected.append((video, quality))
            selected_embeddings.append(video_emb)

            # Stop if reached max_videos
            if max_videos and len(selected) >= max_videos:
                break

        self.selected_videos = [v for v, _ in selected]

        print(f"[Processor] Selected {len(self.selected_videos)} videos")

        return {
            "selected_count": len(self.selected_videos),
            "total_candidates": len(valid_videos),
            "videos": [
                {
                    "path": str(v),
                    "quality": self.quality_scores[v]["final_score"],
                    "technical": self.quality_scores[v]["technical"],
                    "aesthetic": self.quality_scores[v]["aesthetic"],
                    "fastvqa": self.quality_scores[v]["fastvqa"],
                }
                for v in self.selected_videos
            ],
        }

    def get_video_duration(self, video_path):
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(video_path),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
            return float(result.stdout.strip())
        except Exception:
            return None

    def get_video_fps(self, video_path):
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=r_frame_rate",
                    "-of",
                    "csv=p=0",
                    str(video_path),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
            num, den = map(float, result.stdout.strip().split("/"))
            return num / den
        except Exception:
            return None

    def resample_video(self, src_path, dest_path, target_fps=16, video_codec="libx264", preset="fast", crf=23, audio_codec="copy"):
        src_path = Path(src_path)
        dest_path = Path(dest_path)
        target_fps = int(target_fps)  # Ensure integer for ffmpeg

        # Ensure destination directory exists
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Check if FPS is already close to target
        current_fps = self.get_video_fps(src_path)
        if current_fps and abs(current_fps - target_fps) < 0.5:
            shutil.copy2(src_path, dest_path)
            return True

        # Resample using ffmpeg
        try:
            # First check if source file exists and is readable
            if not src_path.exists():
                print(f"[ERROR] Source file does not exist: {src_path}")
                return False

            # Get video info to validate the file
            current_fps = self.get_video_fps(src_path)

            # Convert paths to proper format for FFmpeg on Windows
            src_path_str = str(src_path)
            dest_path_str = str(dest_path)

            # On Windows, ensure paths are properly escaped
            import sys
            if sys.platform == "win32":
                src_path_str = src_path_str.replace("\\", "/")
                dest_path_str = dest_path_str.replace("\\", "/")

            # Build FFmpeg command dynamically
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                src_path_str,
                "-r",
                str(target_fps),
            ]

            # Add video codec settings
            if video_codec != "copy":
                ffmpeg_cmd.extend([
                    "-c:v", video_codec,
                    "-preset", preset,
                    "-crf", str(crf),
                ])
            else:
                ffmpeg_cmd.extend(["-c:v", "copy"])

            # Add audio codec settings
            if audio_codec == "none":
                ffmpeg_cmd.extend(["-an"])  # No audio
            elif audio_codec == "copy":
                ffmpeg_cmd.extend(["-c:a", "copy"])
            else:
                ffmpeg_cmd.extend(["-c:a", audio_codec])

            ffmpeg_cmd.append(dest_path_str)

            result = subprocess.run(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )

            # Verify output file was created
            if dest_path.exists():
                return True
            else:
                print(f"[ERROR] Output file not created: {dest_path}")
                return False

        except subprocess.CalledProcessError as e:
            print(f"[ERROR] FFmpeg failed for {src_path.name}: {e.stderr}")
            return False
        except Exception as e:
            return False

    def copy_selected_videos(
        self, target_dir, resample_fps=None, preserve_structure=False, base_dir=None,
        video_codec="libx264", preset="fast", crf=23, audio_codec="copy"
    ):
        """
        Copy or resample selected videos to target directory.

        Args:
            target_dir: Target directory for copied videos
            resample_fps: If set, resample videos to this FPS
            preserve_structure: Preserve directory structure from base_dir
            base_dir: Base directory for preserving structure
        """
        if not self.selected_videos:
            return {"error": "No videos selected"}

        target_dir = Path(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        results = {"success": [], "failed": [], "total": len(self.selected_videos)}

        for src_path in tqdm(self.selected_videos, desc="Copying videos"):
            try:
                # Determine destination path
                if preserve_structure and base_dir:
                    base_dir = Path(base_dir)
                    rel_path = src_path.relative_to(base_dir)
                    dest_path = target_dir / rel_path
                else:
                    dest_path = target_dir / src_path.name

                # Ensure destination directory exists
                dest_path.parent.mkdir(parents=True, exist_ok=True)

                # Copy or resample
                if resample_fps:
                    success = self.resample_video(src_path, dest_path, resample_fps, video_codec, preset, crf, audio_codec)
                else:
                    shutil.copy2(src_path, dest_path)
                    success = True

                if success:
                    results["success"].append(str(src_path))
                else:
                    results["failed"].append(str(src_path))

            except Exception as e:
                results["failed"].append(str(src_path))

        if results["failed"]:
            print(f"[ERROR] Failed to process {len(results['failed'])} videos")

        return results

    def filter_by_duration(self, min_duration=1.0, max_duration=10.0):
        if not self.selected_videos:
            return {"error": "No videos selected"}

        filtered = []

        for video in tqdm(self.selected_videos, desc="Filtering by duration"):
            duration = self.get_video_duration(video)
            if duration and min_duration <= duration <= max_duration:
                filtered.append(video)

        self.selected_videos = filtered

        return {
            "filtered_count": len(self.selected_videos),
            "min_duration": min_duration,
            "max_duration": max_duration,
        }

    def get_selection_statistics(self):
        if not self.selected_videos:
            return {"error": "No videos selected"}

        stats = {"count": len(self.selected_videos), "videos": []}

        for video in self.selected_videos:
            video_stats = {
                "path": str(video),
                "duration": self.get_video_duration(video),
                "fps": self.get_video_fps(video),
            }

            if video in self.quality_scores:
                quality = self.quality_scores[video]
                video_stats.update(
                    {
                        "quality_score": quality["final_score"],
                        "technical": quality["technical"],
                        "aesthetic": quality["aesthetic"],
                        "fastvqa": quality["fastvqa"],
                    }
                )

            stats["videos"].append(video_stats)

        return stats
