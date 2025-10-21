import gradio as gr
import os
import json
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from tqdm import tqdm

from functions.video_interrogate import init_interrogate_fn
from providers.ovis_16b import OVIS2_MODEL_ID
from providers.qwen_25_vl import QWEN25_VL_MODEL_ID


STYLE_ANALYSIS_PROMPT = """
Analyze this video and identify its distinctive visual style characteristics (5–10).
Focus on these aspects and provide a confidence score (0–1) for each trait you identify.

Examples of visual style traits:
- Color palette (warm/cool tones, saturation, contrast)
- Lighting (natural/artificial, dramatic/soft, color temperature)
- Camera work (static/dynamic, smooth/handheld, angles)
- Composition (rule of thirds, symmetry, depth of field)
- Visual texture (grain, sharpness, blur effects)
- Editing style (fast/slow cuts, transitions, pacing)
- Visual effects (filters, overlays, post-processing)
- Art direction (modern/vintage, minimalist/ornate, realistic/stylized)

Format your response as a JSON object where keys are specific style trait names and values are confidence scores between 0 and 1.
Example: {"warm_color_palette": 0.8, "handheld_camera": 0.6, "fast_editing": 0.9}

{0}
"""


def create_analysis_prompt(
    default_prompt_body: str, previous_prompt_result: str | None
) -> str:
    if previous_prompt_result:
        return f"""Previous analysis results (for context only, update or refine if needed):
{previous_prompt_result}

{default_prompt_body}"""
    else:
        return default_prompt_body


class VideoSelector:
    def __init__(self, kubin, state):
        self.kubin = kubin
        self.state = state

        self.current_model = {
            "model": None,
            "tokenizer": None,
            "name": None,
            "fn": None,
        }
        self.video_analysis_results = {}
        self.selected_videos = []

    def get_video_paths(self, video_dir: str, extensions: List[str]) -> List[str]:
        video_paths = []
        path_obj = Path(video_dir)

        if not path_obj.exists():
            return []

        for ext in extensions:
            video_paths.extend(list(path_obj.glob(f"*{ext}")))

        return [str(path) for path in video_paths]

    def parse_style_traits(self, vlm_response: str) -> Dict[str, float]:
        try:
            if vlm_response.strip().startswith("{"):
                return json.loads(vlm_response)

            traits = {}
            lines = vlm_response.split("\n")

            for line in lines:
                if ":" in line and any(char.isdigit() for char in line):
                    parts = line.split(":")
                    if len(parts) == 2:
                        trait_name = parts[0].strip().lower().replace(" ", "_")
                        try:
                            score_text = parts[1].strip()
                            score = float(
                                "".join(
                                    c for c in score_text if c.isdigit() or c == "."
                                )
                            )
                            if 0 <= score <= 1:
                                traits[trait_name] = score
                        except:
                            continue
            return traits

        except Exception as e:
            print(f"error parsing VLM response: {e}")
            return {}

    def analyze_videos(
        self,
        prompt: str,
        video_dir: str,
        model_name: str,
        quantization: str,
        use_flash_attention: bool,
        file_extensions: List[str],
        include_subdirectories: bool,
        progress=gr.Progress(),
    ):
        cache_dir = self.kubin.params("general", "cache_dir")
        device = self.kubin.params("general", "device")

        init_interrogate_fn(
            kubin=self.kubin,
            state=self.state,
            cache_dir=cache_dir,
            device=device,
            model_name=model_name,
            quantization=quantization,
            use_flash_attention=use_flash_attention,
        )

        interrogate_fn = self.state["fn"]

        if include_subdirectories:
            video_paths = []
            for root, dirs, files in os.walk(video_dir):
                for file in files:
                    if any(file.endswith(ext) for ext in file_extensions):
                        video_paths.append(os.path.join(root, file))
        else:
            video_paths = self.get_video_paths(video_dir, file_extensions)

        if not video_paths:
            return "No video files found in the specified directory.", None

        progress(0, desc=f"analyzing {len(video_paths)} videos for traits...")

        self.video_analysis_results = {}
        trait_frequency = {}

        previous_response = None
        for i, video_path in enumerate(video_paths):
            progress(
                (i + 1) / len(video_paths),
                desc=f"analyzing video {i+1}/{len(video_paths)}",
            )

            video_id = Path(video_path).stem

            try:
                prompt = create_analysis_prompt(prompt, previous_response)
                vlm_response = interrogate_fn(video_path, prompt)
                traits = self.parse_style_traits(vlm_response)

                if traits:
                    self.video_analysis_results[video_id] = {
                        "path": video_path,
                        "traits": traits,
                        "raw_response": vlm_response,
                    }

                    for trait in traits:
                        trait_frequency[trait] = trait_frequency.get(trait, 0) + 1

            except Exception as e:
                print(f"error analyzing {video_path}: {e}")
                continue

        analysis_summary = f"analysis complete!\n"
        analysis_summary += f"- analyzed {len(video_paths)} videos\n"
        analysis_summary += (
            f"- successfully processed {len(self.video_analysis_results)} videos\n"
        )
        analysis_summary += f"- found {len(trait_frequency)} unique traits\n"

        results_data = []
        for video_id, data in self.video_analysis_results.items():
            traits_str = ", ".join(
                [f"{k}: {v:.2f}" for k, v in list(data["traits"].items())[:5]]
            )
            results_data.append(
                {
                    "video id": video_id,
                    "path": data["path"],
                    "top traits": traits_str,
                    "total traits": len(data["traits"]),
                }
            )

        results_df = pd.DataFrame(results_data)
        return analysis_summary, results_df

    def select_best_videos(self, num_videos: int = 10):
        if not self.video_analysis_results:
            return "no analyzed videos available, please analyze videos first.", None

        trait_frequency = {}
        for video_data in self.video_analysis_results.values():
            for trait in video_data["traits"]:
                trait_frequency[trait] = trait_frequency.get(trait, 0) + 1

        total_videos = len(self.video_analysis_results)
        important_traits = []

        for trait, freq in trait_frequency.items():
            if 0.05 * total_videos <= freq <= 0.5 * total_videos:
                important_traits.append(trait)

        video_scores = {}
        for video_id, video_data in self.video_analysis_results.items():
            traits = video_data["traits"]

            trait_count = sum(1 for t in important_traits if t in traits)

            trait_strength = sum(
                traits.get(t, 0) for t in important_traits if t in traits
            )
            trait_strength = trait_strength / trait_count if trait_count > 0 else 0
            coverage_score = trait_count / max(1, len(important_traits))
            video_scores[video_id] = 0.6 * coverage_score + 0.4 * trait_strength

        selected = []
        remaining = list(video_scores.keys())

        if remaining:
            first_video = max(remaining, key=lambda v: video_scores[v])
            selected.append(first_video)
            remaining.remove(first_video)

        while len(selected) < num_videos and remaining:
            best_video = None
            best_score = -1

            covered_traits = set()
            for vid in selected:
                covered_traits.update(self.video_analysis_results[vid]["traits"].keys())

            for video_id in remaining:
                video_traits = set(
                    self.video_analysis_results[video_id]["traits"].keys()
                )
                new_traits = video_traits - covered_traits

                uniqueness_score = len(new_traits) / max(1, len(video_traits))
                quality_score = video_scores[video_id]
                combined_score = 0.7 * uniqueness_score + 0.3 * quality_score

                if combined_score > best_score:
                    best_score = combined_score
                    best_video = video_id

            if best_video:
                selected.append(best_video)
                remaining.remove(best_video)
            else:
                break

        self.selected_videos = []
        for video_id in selected:
            video_data = self.video_analysis_results[video_id]
            traits = video_data["traits"]
            top_traits = dict(
                sorted(traits.items(), key=lambda x: x[1], reverse=True)[:5]
            )

            self.selected_videos.append(
                {
                    "video_id": video_id,
                    "path": video_data["path"],
                    "score": video_scores.get(video_id, 0),
                    "top_traits": top_traits,
                }
            )

        self.selected_videos.sort(key=lambda x: x["score"], reverse=True)

        summary = f"selected {len(self.selected_videos)} videos for style training:\n\n"
        for i, video in enumerate(self.selected_videos, 1):
            summary += f"{i}. {video['video_id']} (score: {video['score']:.3f})\n"
            summary += f"   top traits: {', '.join(video['top_traits'].keys())}\n\n"

        df_data = []
        for i, video in enumerate(self.selected_videos, 1):
            df_data.append(
                {
                    "rank": i,
                    "video id": video["video_id"],
                    "path": video["path"],
                    "score": f"{video['score']:.3f}",
                    "top traits": ", ".join(
                        [f"{k}: {v:.2f}" for k, v in video["top_traits"].items()]
                    ),
                }
            )

        results_df = pd.DataFrame(df_data)
        return summary, results_df

    def export_selected_videos(self, output_file: str):
        if not self.selected_videos:
            return "No videos selected. Please run video selection first."

        try:
            export_data = {
                "selected_videos": self.selected_videos,
                "total_analyzed": len(self.video_analysis_results),
                "selection_criteria": "Style trait diversity and strength",
                "important_traits_threshold": "5-50% frequency",
            }

            with open(output_file, "w") as f:
                json.dump(export_data, f, indent=2)

            return f"successfully exported {len(self.selected_videos)} selected videos to {output_file}"

        except Exception as e:
            return f"error exporting videos: {str(e)}"
