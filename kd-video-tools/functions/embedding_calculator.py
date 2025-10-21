import torch
import numpy as np
import cv2
import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from transformers import XCLIPProcessor, XCLIPModel


class EmbeddingCalculator:
    def __init__(self, kubin):
        self.kubin = kubin
        self.model = None
        self.processor = None
        self.device = None
        self.embeddings_cache = {}

    def load_model(self, model_name="microsoft/xclip-base-patch32", device="cuda"):
        print(f"[X-CLIP] Loading model: {model_name}")

        self.device = device
        self.model = XCLIPModel.from_pretrained(model_name)
        self.model = self.model.to(device)
        self.model.eval()
        self.model.config.return_dict = True

        self.processor = XCLIPProcessor.from_pretrained(model_name)

        print(f"[X-CLIP] Model loaded on {device}")
        return True

    def extract_frames(self, video_path, num_frames=8):
        num_frames = int(num_frames)
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            cap.release()
            return None

        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames = []

        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        cap.release()
        return frames if frames else None

    @torch.no_grad()
    def compute_embeddings_batch(self, frames_list):
        if not self.model or not self.processor:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        valid_videos = [
            (i, frames) for i, frames in enumerate(frames_list) if frames is not None
        ]
        if not valid_videos:
            return [None] * len(frames_list)

        indices, valid_frames = zip(*valid_videos)

        batch_size = len(valid_frames)
        num_frames = len(valid_frames[0])

        all_frames = []
        for frames in valid_frames:
            for frame in frames:
                all_frames.append(Image.fromarray(frame))

        # Process all frames
        inputs = self.processor.image_processor(images=all_frames, return_tensors="pt")
        pixel_values = inputs["pixel_values"]

        # Reshape to [batch_size, num_frames, C, H, W]
        pixel_values = pixel_values.squeeze(0)
        C, H, W = pixel_values.shape[1], pixel_values.shape[2], pixel_values.shape[3]
        pixel_values = pixel_values.reshape(batch_size, num_frames, C, H, W)
        pixel_values = pixel_values.to(self.device)

        # Compute embeddings
        batch_size_actual, num_frames_actual = (
            pixel_values.shape[0],
            pixel_values.shape[1],
        )
        pixel_values_reshaped = pixel_values.reshape(-1, C, H, W)

        vision_outputs = self.model.vision_model(
            pixel_values=pixel_values_reshaped,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )

        frame_embeds = vision_outputs.pooler_output
        frame_embeds = self.model.visual_projection(frame_embeds)

        cls_features = frame_embeds.view(batch_size_actual, num_frames_actual, -1)

        mit_outputs = self.model.mit(
            cls_features,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )

        if isinstance(mit_outputs, tuple):
            video_embeds = mit_outputs[1]
        else:
            video_embeds = mit_outputs.pooler_output

        embeddings = video_embeds / video_embeds.norm(dim=-1, keepdim=True)
        embeddings = embeddings.cpu().numpy()

        result = [None] * len(frames_list)
        for idx, emb in zip(indices, embeddings):
            result[idx] = emb.tolist()

        return result

    def calculate_embeddings(
        self,
        video_dir,
        video_extensions=[".mp4", ".avi", ".mov", ".mkv", ".webm"],
        num_frames=8,
        batch_size=4,
        cache_file=None,
        include_subdirectories=False,
        progress_callback=None,
    ):
        video_dir = Path(video_dir)

        if not video_dir.exists():
            return {"error": f"Directory {video_dir} does not exist"}

        # Find all videos
        if include_subdirectories:
            all_videos = []
            for ext in video_extensions:
                all_videos.extend(video_dir.rglob(f"*{ext}"))
        else:
            all_videos = []
            for ext in video_extensions:
                all_videos.extend(video_dir.glob(f"*{ext}"))

        print(f"Found {len(all_videos)} videos")

        if cache_file and Path(cache_file).exists():
            with open(cache_file, "r") as f:
                self.embeddings_cache = {
                    Path(k).resolve(): v for k, v in json.load(f).items()
                }
            print(f"Loaded {len(self.embeddings_cache)} cached embeddings")

        videos_to_process = [
            v for v in all_videos if v.resolve() not in self.embeddings_cache
        ]

        print(
            f"Processing {len(videos_to_process)} new videos, using cache for {len(all_videos) - len(videos_to_process)}"
        )

        with ThreadPoolExecutor(max_workers=1) as executor:
            for i in tqdm(
                range(0, len(videos_to_process), batch_size),
                desc="Computing embeddings",
            ):
                batch = videos_to_process[i : i + batch_size]
                frames_batch = list(
                    executor.map(lambda v: self.extract_frames(v, num_frames), batch)
                )

                try:
                    emb_batch = self.compute_embeddings_batch(frames_batch)

                    for video_path, emb in zip(batch, emb_batch):
                        if emb is not None:
                            self.embeddings_cache[video_path.resolve()] = emb

                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        torch.cuda.empty_cache()
                        continue
                    raise e

        # Save cache
        if cache_file:
            with open(cache_file, "w") as f:
                json.dump({str(k): v for k, v in self.embeddings_cache.items()}, f)
            print(f"Saved embeddings cache to {cache_file}")

        return {
            "total_videos": len(all_videos),
            "processed": len(videos_to_process),
            "cached": len(all_videos) - len(videos_to_process),
            "embeddings": self.embeddings_cache,
        }

    def save_embedding_file(
        self, video_path, embedding, extension=".embedding", model_info=None
    ):
        video_path = Path(video_path)
        embedding_file = video_path.parent / (video_path.name + extension)

        data = {
            "embedding": embedding,
            "model": model_info or "microsoft/xclip-base-patch32",
            "dimensions": len(embedding) if embedding else 0,
        }

        with open(embedding_file, "w", encoding="utf-8") as f:
            json.dump(data, f)

        return embedding_file

    def load_embedding_file(self, video_path, extension=".embedding"):
        video_path = Path(video_path)
        embedding_file = video_path.parent / (video_path.name + extension)

        if not embedding_file.exists():
            return None

        with open(embedding_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("embedding")

    def calculate_embeddings_per_file(
        self,
        video_dir,
        video_extensions=[".mp4", ".avi", ".mov", ".mkv", ".webm"],
        num_frames=8,
        batch_size=4,
        include_subdirectories=False,
        embedding_extension=".embedding",
        overwrite_existing=False,
        model_name="microsoft/xclip-base-patch32",
    ):
        """Calculate embeddings for videos and save as .embedding files."""
        num_frames = int(num_frames)
        batch_size = int(batch_size)
        video_dir = Path(video_dir)

        if not video_dir.exists():
            return {"error": f"Directory {video_dir} does not exist"}

        if include_subdirectories:
            all_videos = []
            for ext in video_extensions:
                all_videos.extend(video_dir.rglob(f"*{ext}"))
        else:
            all_videos = []
            for ext in video_extensions:
                all_videos.extend(video_dir.glob(f"*{ext}"))

        print(f"Found {len(all_videos)} videos")

        videos_to_process = []
        for video in all_videos:
            embedding_file = video.parent / (video.name + embedding_extension)
            if overwrite_existing or not embedding_file.exists():
                videos_to_process.append(video)

        print(
            f"Processing {len(videos_to_process)} videos ({len(all_videos) - len(videos_to_process)} already have .embedding files)"
        )

        processed_count = 0
        failed_count = 0

        with ThreadPoolExecutor(max_workers=1) as executor:
            for i in tqdm(
                range(0, len(videos_to_process), batch_size),
                desc="Computing embeddings",
            ):
                batch = videos_to_process[i : i + batch_size]
                frames_batch = list(
                    executor.map(lambda v: self.extract_frames(v, num_frames), batch)
                )

                try:
                    emb_batch = self.compute_embeddings_batch(frames_batch)

                    for video_path, emb in zip(batch, emb_batch):
                        if emb is not None:
                            self.save_embedding_file(
                                video_path, emb, embedding_extension, model_name
                            )
                            processed_count += 1
                        else:
                            failed_count += 1

                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        torch.cuda.empty_cache()
                        failed_count += len(batch)
                        continue
                    raise e

        return {
            "total_videos": len(all_videos),
            "processed": processed_count,
            "skipped": len(all_videos) - len(videos_to_process),
            "failed": failed_count,
        }

    def unload_model(self):
        """Unload model to free memory."""
        if self.model:
            self.model = self.model.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("[X-CLIP] Model unloaded from GPU")
