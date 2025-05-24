import os
import json
import shutil
from pathlib import Path

DEFAULT_CLASSIFIER_TEMPLATE = """Analyze this video and classify it based on its content. Respond with a JSON object:
{
  "category": <number>,
  "caption": "<string>"
}
CATEGORIES:
- 2: Predominantly fast-paced content
- 1: Predominantly slow-paced content  
- 0: Not certain
CAPTION: Describe main visual content in 20-30 words.
Return only JSON, no additional text.
"""


def classify(filepath: str, vlm_output: str, base_dir) -> bool:
    video_path = Path(filepath)
    base_dir = Path(base_dir)

    try:
        classification = json.loads(vlm_output.strip())
    except json.JSONDecodeError:
        start_idx = vlm_output.find("{")
        end_idx = vlm_output.rfind("}")

        if start_idx != -1 and end_idx != -1:
            json_str = vlm_output[start_idx : end_idx + 1]
            classification = json.loads(json_str)
        else:
            print(f"error: could not parse JSON from VLM output: {vlm_output}")
            return False

    category = classification.get("category", "-")
    caption = classification.get("caption", None)

    category_folder = base_dir / str(category)
    category_folder.mkdir(parents=True, exist_ok=True)

    dest_video_path = category_folder / video_path.name
    shutil.copy2(video_path, dest_video_path)

    if caption is not None:
        caption_file = category_folder / f"{video_path.stem}.txt"
        with open(caption_file, "w", encoding="utf-8") as f:
            f.write(caption)
