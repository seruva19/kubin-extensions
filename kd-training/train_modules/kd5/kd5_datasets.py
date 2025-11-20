"""Dataset helpers and caching utilities for KD5 training."""

from __future__ import annotations

import gc
import json
import math
import random
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler
from tqdm.auto import tqdm

from .kd5_config import (
    KD5AspectBucket,
    KD5BucketSettings,
    KD5DatasetConfig,
    ensure_dirs,
)


def _bucket_key(width: int, height: int) -> str:
    return f"{width}x{height}"


def _dedupe_buckets(buckets: List[KD5AspectBucket]) -> List[KD5AspectBucket]:
    unique = {}
    for bucket in buckets or []:
        key = _bucket_key(bucket.width, bucket.height)
        if key not in unique:
            unique[key] = KD5AspectBucket(width=bucket.width, height=bucket.height)
    return list(unique.values())


def _select_bucket(
    buckets: List[KD5AspectBucket], width: int, height: int
) -> KD5AspectBucket:
    if not buckets:
        return KD5AspectBucket(width=width, height=height)
    safe_height = max(height, 1)
    target_ratio = width / safe_height
    best = buckets[0]
    best_diff = abs(target_ratio - best.width / max(best.height, 1))
    for candidate in buckets[1:]:
        diff = abs(target_ratio - candidate.width / max(candidate.height, 1))
        if diff < best_diff:
            best = candidate
            best_diff = diff
    return best


def _resize_to_bucket(
    frame: np.ndarray, target_width: int, target_height: int
) -> np.ndarray:
    src_height, src_width = frame.shape[:2]
    if src_height <= 0 or src_width <= 0:
        return np.zeros(
            (target_height, target_width, frame.shape[2] if frame.ndim == 3 else 1),
            dtype=frame.dtype,
        )
    scale = max(target_width / src_width, target_height / src_height)
    new_width = max(1, int(round(src_width * scale)))
    new_height = max(1, int(round(src_height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(frame, (new_width, new_height), interpolation=interpolation)
    if new_width > target_width:
        start = (new_width - target_width) // 2
        resized = resized[:, start : start + target_width]
    elif new_width < target_width:
        pad_left = (target_width - new_width) // 2
        pad_right = target_width - new_width - pad_left
        resized = cv2.copyMakeBorder(
            resized, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
    if new_height > target_height:
        start = (new_height - target_height) // 2
        resized = resized[start : start + target_height, :]
    elif new_height < target_height:
        pad_top = (target_height - new_height) // 2
        pad_bottom = target_height - new_height - pad_top
        resized = cv2.copyMakeBorder(
            resized, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
    return resized[:target_height, :target_width]


def _coerce_positive_int(value, default=None):
    if value in (None, ""):
        return default
    try:
        value_int = int(value)
    except (TypeError, ValueError):
        return default
    return value_int if value_int > 0 else default


def _generate_auto_buckets(
    settings: KD5BucketSettings, base_bucket: KD5AspectBucket, *, patch_size: int = 8
) -> List[KD5AspectBucket]:
    min_size = max(int(settings.min_size or 1), 1)
    max_width = max(int(settings.max_width or min_size), min_size)
    max_height = max(int(settings.max_height or min_size), min_size)
    divisible = max(int(settings.divisible or 1), 1)
    step = max(int(settings.step_size or divisible), 1)

    def _token_span(value: int) -> int:
        return max(1, (value + patch_size - 1) // patch_size)

    max_tokens = settings.max_tokens
    if max_tokens in (None, ""):
        max_tokens = _token_span(max_width) * _token_span(max_height)
    else:
        max_tokens = max(int(max_tokens), 1)

    candidates = set()
    for width in range(min_size, max_width + 1, step):
        if width % divisible != 0:
            continue
        for height in range(min_size, max_height + 1, step):
            if height % divisible != 0:
                continue
            tokens = _token_span(width) * _token_span(height)
            if tokens <= max_tokens:
                candidates.add((width, height))

    buckets = [
        KD5AspectBucket(width=w, height=h)
        for w, h in sorted(candidates, key=lambda item: (item[0] * item[1], item[0]))
    ]
    if settings.include_base_resolution:
        buckets.append(base_bucket)
    buckets = _dedupe_buckets(buckets)
    if not buckets:
        buckets = [base_bucket]
    return buckets


def _find_best_bucket(
    registry: Dict[str, KD5AspectBucket], aspect_ratio: float
) -> Tuple[str, float]:
    ratio = max(aspect_ratio, 1e-6)
    best_key = None
    best_rel = float("inf")
    for key, bucket in registry.items():
        bucket_ratio = bucket.width / max(bucket.height, 1)
        rel = abs(bucket_ratio - ratio) / max(bucket_ratio, 1e-6)
        if rel < best_rel:
            best_rel = rel
            best_key = key
    return best_key, best_rel


class KD5BucketPlanner:
    def __init__(self, cfg: KD5DatasetConfig, samples: List[Dict]):
        self.cfg = cfg
        self.samples = samples
        self.settings = cfg.bucket_settings

    def plan(
        self,
    ) -> Tuple[Dict[str, KD5AspectBucket], List[str], Dict[str, object], str]:
        mode = (self.settings.mode or "auto").lower()
        base_bucket = KD5AspectBucket(
            width=self.cfg.resolution, height=self.cfg.resolution
        )
        fallback_bucket = self.settings.fallback or base_bucket

        if mode == "manual" and self.cfg.aspect_buckets:
            buckets = _dedupe_buckets(self.cfg.aspect_buckets + [fallback_bucket])
        else:
            buckets = _generate_auto_buckets(self.settings, base_bucket)
            buckets.append(fallback_bucket)
            buckets = _dedupe_buckets(buckets)
            mode = "auto"

        registry: Dict[str, KD5AspectBucket] = {
            _bucket_key(bucket.width, bucket.height): bucket for bucket in buckets
        }
        fallback_key = _bucket_key(fallback_bucket.width, fallback_bucket.height)
        registry[fallback_key] = fallback_bucket

        bucket_keys: List[str] = []
        tolerance_hits = 0
        max_ar_error = max(float(self.settings.max_ar_error or 0.0), 0.0)

        for sample in self.samples:
            width = _coerce_positive_int(
                sample.get("orig_width") or sample.get("width"), fallback_bucket.width
            )
            height = _coerce_positive_int(
                sample.get("orig_height") or sample.get("height"),
                fallback_bucket.height,
            )
            safe_height = max(height, 1)
            ratio = width / safe_height
            best_key, rel_diff = _find_best_bucket(registry, ratio)
            if best_key is None:
                best_key = fallback_key
                rel_diff = float("inf")
            if rel_diff > max_ar_error:
                tolerance_hits += 1
                bucket_keys.append(fallback_key)
            else:
                bucket_keys.append(best_key)

        bucket_counts: Dict[str, int] = defaultdict(int)
        for key in bucket_keys:
            bucket_counts[key] += 1

        if self.settings.max_buckets and len(bucket_counts) > int(
            self.settings.max_buckets
        ):
            sorted_keys = sorted(
                bucket_counts.items(), key=lambda item: item[1], reverse=True
            )
            allowed = {key for key, _ in sorted_keys[: int(self.settings.max_buckets)]}
            allowed.add(fallback_key)
            for idx, key in enumerate(bucket_keys):
                if key not in allowed:
                    bucket_keys[idx] = fallback_key
            bucket_counts = defaultdict(int)
            for key in bucket_keys:
                bucket_counts[key] += 1
            registry = {key: registry[key] for key in allowed if key in registry}
            registry[fallback_key] = fallback_bucket

        diagnostics = {
            "mode": mode,
            "total_samples": len(self.samples),
            "bucket_counts": dict(
                sorted(bucket_counts.items(), key=lambda item: item[1], reverse=True)
            ),
            "tolerance_hits": tolerance_hits,
            "generated_bucket_count": len(registry),
        }

        return registry, bucket_keys, diagnostics, fallback_key


class KD5RawDataset(Dataset):
    def __init__(self, cfg: KD5DatasetConfig):
        self.cfg = cfg
        if not cfg.raw_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {cfg.raw_path}")
        samples = self._load_metadata()
        planner = KD5BucketPlanner(cfg, samples)
        bucket_registry, assignments, diagnostics, fallback_key = planner.plan()
        for sample, bucket_key in zip(samples, assignments):
            bucket = bucket_registry[bucket_key]
            sample["bucket"] = {"width": bucket.width, "height": bucket.height}
            sample["bucket_key"] = bucket_key

        self.metadata = samples
        self.bucket_registry = bucket_registry
        self.bucket_keys = assignments
        self.bucket_diagnostics = diagnostics
        self.default_bucket_key = fallback_key
        self.buckets = list(bucket_registry.values())

    def _load_metadata(self) -> List[Dict]:
        metadata_path = self.cfg.raw_path / "metadata.json"
        video_exts = {".mp4", ".webm", ".mov", ".avi", ".mkv"}
        image_exts = {".png", ".jpg", ".jpeg", ".webp"}

        raw_entries: List[Dict] = []
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            raw_entries = data.get("samples", []) or []
        else:
            for file in sorted(self.cfg.raw_path.rglob("*")):
                if not file.is_file():
                    continue
                suffix = file.suffix.lower()
                media_type = None
                if suffix in video_exts:
                    media_type = "video"
                elif suffix in image_exts:
                    media_type = "image"
                if not media_type:
                    continue
                caption_path = file.with_suffix(".txt")
                caption = (
                    caption_path.read_text(encoding="utf-8").strip()
                    if caption_path.exists()
                    else ""
                )
                raw_entries.append(
                    {
                        "path": file.relative_to(self.cfg.raw_path).as_posix(),
                        "caption": caption,
                        "type": media_type,
                    }
                )

        samples: List[Dict] = []
        for entry in raw_entries:
            if not isinstance(entry, dict):
                try:
                    entry = dict(entry)
                except Exception:
                    continue
            rel_path = entry.get("path")
            if not rel_path:
                continue
            media_path = (self.cfg.raw_path / rel_path).resolve()
            if not media_path.exists():
                continue
            media_type = entry.get("type")
            if media_type not in {"image", "video"}:
                suffix = media_path.suffix.lower()
                if suffix in video_exts:
                    media_type = "video"
                else:
                    media_type = "image"
            width, height = self._probe_dimensions(media_path, media_type)
            entry["type"] = media_type
            entry["orig_width"] = width
            entry["orig_height"] = height
            entry["caption"] = entry.get("caption", "") or ""
            samples.append(entry)
        return samples

    def _probe_dimensions(self, path: Path, media_type: str) -> Tuple[int, int]:
        width = 0
        height = 0
        if media_type == "video":
            capture = cv2.VideoCapture(str(path))
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
            capture.release()
        else:
            try:
                with Image.open(path) as image:
                    width, height = image.size
            except Exception:
                width, height = 0, 0
        if width <= 0 or height <= 0:
            fallback = max(1, int(self.cfg.resolution))
            return fallback, fallback
        return int(width), int(height)

    def __len__(self) -> int:
        return len(self.metadata)

    def _load_video(
        self, path: Path, target_width: int, target_height: int
    ) -> torch.Tensor:
        frames_needed = self.cfg.frames
        capture = cv2.VideoCapture(str(path))
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        indices = (
            np.linspace(0, total_frames - 1, frames_needed, dtype=int)
            if total_frames >= frames_needed
            else list(range(total_frames))
        )
        frames = []
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            success, frame = capture.read()
            if not success:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = _resize_to_bucket(frame, target_width, target_height)
            tensor = torch.from_numpy(frame).float() / 255.0
            frames.append(tensor.permute(2, 0, 1))
        capture.release()
        if not frames:
            raise RuntimeError(f"Could not decode frames from {path}")
        video = torch.stack(frames, dim=0)
        if video.shape[0] < frames_needed:
            repeat_count = frames_needed - video.shape[0]
            video = torch.cat([video, video[-1:].repeat(repeat_count, 1, 1, 1)], dim=0)
        video = video * 2.0 - 1.0
        return video

    def _load_image(
        self, path: Path, target_width: int, target_height: int
    ) -> torch.Tensor:
        with Image.open(path) as img:
            image = img.convert("RGB")
            array = np.array(image)
        array = _resize_to_bucket(array, target_width, target_height)
        tensor = torch.from_numpy(array).float() / 255.0
        tensor = tensor.permute(2, 0, 1)
        tensor = tensor.unsqueeze(0).repeat(self.cfg.frames, 1, 1, 1)
        tensor = tensor * 2.0 - 1.0
        return tensor

    def __getitem__(self, index: int) -> Dict:
        sample = self.metadata[index]
        media_path = self.cfg.raw_path / sample["path"]
        bucket_info = sample.get("bucket") or {}
        target_width = int(bucket_info.get("width", self.cfg.resolution))
        target_height = int(bucket_info.get("height", self.cfg.resolution))
        if sample.get("type") == "video":
            pixel_values = self._load_video(media_path, target_width, target_height)
        else:
            pixel_values = self._load_image(media_path, target_width, target_height)
        frames = pixel_values.shape[0]
        return {
            "pixel_values": pixel_values,
            "caption": sample.get("caption", ""),
            "frames": frames,
            "type": sample.get("type", "image"),
            "dataset": self.cfg.name,
            "bucket": {"width": target_width, "height": target_height},
            "bucket_key": sample.get(
                "bucket_key", _bucket_key(target_width, target_height)
            ),
        }


class KD5CachedDataset(Dataset):
    def __init__(self, cache_dir: Path, dataset_cfg: KD5DatasetConfig):
        self.cache_dir = cache_dir
        self.dataset_cfg = dataset_cfg
        self.dataset_name = dataset_cfg.name
        metadata_path = cache_dir / "cache_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Cache metadata not found at {metadata_path}. Run cache builder for {self.dataset_name}."
            )
        with open(metadata_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)

        registry_data = data.get("bucket_registry") or {}
        bucket_registry: Dict[str, KD5AspectBucket] = {}
        for key, dims in registry_data.items():
            if not isinstance(dims, dict):
                continue
            try:
                width = int(dims.get("width"))
                height = int(dims.get("height"))
            except (TypeError, ValueError):
                continue
            if width and height:
                bucket_registry[key] = KD5AspectBucket(width=width, height=height)

        raw_entries = data.get("samples", []) or []
        self.entries: List[Dict] = []
        for entry in raw_entries:
            if not isinstance(entry, dict):
                entry = dict(entry)
            prepared = self._prepare_entry(entry)
            if prepared is not None:
                bucket_info = prepared.get("bucket") or {}
                bucket_key = prepared.get("bucket_key")
                if bucket_key and bucket_key not in bucket_registry:
                    bucket_registry[bucket_key] = KD5AspectBucket(
                        width=int(
                            bucket_info.get("width", self.dataset_cfg.resolution)
                        ),
                        height=int(
                            bucket_info.get("height", self.dataset_cfg.resolution)
                        ),
                    )
                self.entries.append(prepared)

        self.bucket_registry = (
            bucket_registry
            if bucket_registry
            else {
                _bucket_key(
                    self.dataset_cfg.resolution, self.dataset_cfg.resolution
                ): KD5AspectBucket(
                    width=self.dataset_cfg.resolution,
                    height=self.dataset_cfg.resolution,
                )
            }
        )
        self.bucket_keys = [entry["bucket_key"] for entry in self.entries]
        self.default_bucket_key = (
            data.get("default_bucket_key")
            if isinstance(data.get("default_bucket_key"), str)
            else (
                self.bucket_keys[0]
                if self.bucket_keys
                else next(iter(self.bucket_registry))
            )
        )
        if self.default_bucket_key not in self.bucket_registry:
            bucket = self.bucket_registry[next(iter(self.bucket_registry))]
            self.default_bucket_key = _bucket_key(bucket.width, bucket.height)
        self.bucket_diagnostics = data.get("bucket_diagnostics", {})
        self.buckets = list(self.bucket_registry.values())
        self.latent_dir = cache_dir / "latents"
        self.text_dir = cache_dir / "text_embeddings"

    def _prepare_entry(self, entry: Dict) -> Optional[Dict]:
        cache_key = entry.get("cache_key")
        if not cache_key:
            return None
        bucket_info = (
            entry.get("bucket") if isinstance(entry.get("bucket"), dict) else None
        )
        width = _coerce_positive_int(bucket_info.get("width")) if bucket_info else None
        height = (
            _coerce_positive_int(bucket_info.get("height")) if bucket_info else None
        )
        if width is None or height is None:
            width = _coerce_positive_int(entry.get("orig_width") or entry.get("width"))
            height = _coerce_positive_int(
                entry.get("orig_height") or entry.get("height")
            )
        if width is None or height is None:
            width = self.dataset_cfg.resolution
            height = self.dataset_cfg.resolution
        width = int(width)
        height = int(height)
        bucket_key = entry.get("bucket_key")
        if bucket_key in (None, ""):
            bucket_key = _bucket_key(width, height)
        entry["bucket"] = {"width": width, "height": height}
        entry["bucket_key"] = bucket_key
        entry.setdefault("caption", "")
        return entry

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> Dict:
        entry = self.entries[index]
        cache_key = entry["cache_key"]
        latent_path = self.latent_dir / f"{cache_key}.pt"
        text_path = self.text_dir / f"{cache_key}.pt"
        if not latent_path.exists():
            raise FileNotFoundError(
                f"Latents missing for key {cache_key} (dataset {self.dataset_name})"
            )
        if not text_path.exists():
            raise FileNotFoundError(
                f"Text embeddings missing for key {cache_key} (dataset {self.dataset_name})"
            )
        latents = torch.load(latent_path, map_location="cpu")
        text_embeddings = torch.load(text_path, map_location="cpu")
        if "text_embeds" not in text_embeddings or "cu_seqlens" not in text_embeddings:
            raise ValueError(
                f"Corrupt text embeddings for {cache_key}; rebuild cache for {self.dataset_name}."
            )
        bucket_key = entry.get("bucket_key") or self.default_bucket_key
        bucket_info = entry.get("bucket")
        if not bucket_info:
            bucket = (
                self.bucket_registry.get(bucket_key)
                or self.bucket_registry[self.default_bucket_key]
            )
            bucket_info = {"width": bucket.width, "height": bucket.height}
        bucket = self.bucket_registry.get(bucket_key)
        if bucket is None and bucket_info:
            bucket = KD5AspectBucket(
                width=int(bucket_info.get("width", self.dataset_cfg.resolution)),
                height=int(bucket_info.get("height", self.dataset_cfg.resolution)),
            )
            self.bucket_registry[bucket_key] = bucket
        elif bucket is None:
            bucket = self.bucket_registry[self.default_bucket_key]
            bucket_info = {"width": bucket.width, "height": bucket.height}
            bucket_key = self.default_bucket_key
        return {
            "latents": latents,
            "text_embeddings": text_embeddings,
            "caption": entry.get("caption", ""),
            "frames": entry.get("frames", latents.shape[0]),
            "dataset": self.dataset_name,
            "bucket": bucket_info,
            "bucket_key": bucket_key,
        }


def collate_cached_batch(samples: List[Dict]) -> Dict:
    latents = torch.stack([sample["latents"] for sample in samples], dim=0)
    captions = [sample["caption"] for sample in samples]
    frames = torch.tensor([sample["frames"] for sample in samples], dtype=torch.int64)
    text_embed_list = [
        sample["text_embeddings"]["text_embeds"] for sample in samples
    ]
    text_embeds = torch.cat(text_embed_list, dim=0)
    pooled = torch.stack(
        [sample["text_embeddings"]["pooled_embed"] for sample in samples], dim=0
    )
    cu_parts = [torch.tensor([0], dtype=torch.int32)]
    offset = 0
    for sample in samples:
        cu = sample["text_embeddings"]["cu_seqlens"].to(dtype=torch.int32)
        cu_shifted = cu + offset
        offset = int(cu_shifted[-1].item())
        cu_parts.append(cu_shifted[1:])
    cu_seqlens = torch.cat(cu_parts, dim=0)
    dataset_names = [sample["dataset"] for sample in samples]
    bucket_info = [sample.get("bucket") for sample in samples]
    bucket_keys = [sample.get("bucket_key") for sample in samples]
    return {
        "latents": latents,
        "text_embeds": text_embeds,
        "pooled_embed": pooled,
        "cu_seqlens": cu_seqlens,
        "caption": captions,
        "frames": frames,
        "dataset": dataset_names,
        "bucket": bucket_info,
        "bucket_key": bucket_keys,
    }


class AutoBucketBatchSampler(Sampler[List[List[int]]]):
    def __init__(
        self,
        bucket_assignments: Dict[int, str],
        sample_weights: Dict[int, float],
        *,
        batch_size: int,
        allow_partial: bool = True,
        drop_last: bool = False,
        seed: int = 42,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.bucket_assignments = bucket_assignments
        self.sample_weights = sample_weights
        self.batch_size = batch_size
        self.allow_partial = allow_partial
        self.drop_last = drop_last
        self.seed = seed
        self._current_plan: Optional[List[List[int]]] = None
        self._plan_metadata: Optional[List[Dict[str, object]]] = None
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = max(int(epoch), 0)
        self._current_plan = None
        self._plan_metadata = None

    @property
    def plan_metadata(self) -> List[Dict[str, object]]:
        return list(self._plan_metadata or [])

    def __len__(self) -> int:
        self._ensure_plan()
        return len(self._current_plan or [])

    def __iter__(self):
        self._ensure_plan()
        plan = self._current_plan or []
        for batch in plan:
            yield batch
        self._epoch += 1
        self._current_plan = None

    def _ensure_plan(self) -> None:
        if self._current_plan is None:
            plan, metadata = self._build_plan(self.seed + self._epoch)
            self._current_plan = plan
            self._plan_metadata = metadata

    def _build_plan(
        self, epoch_seed: int
    ) -> Tuple[List[List[int]], List[Dict[str, object]]]:
        if not self.bucket_assignments:
            return [], []
        rng = random.Random(epoch_seed)
        buckets: Dict[str, List[int]] = defaultdict(list)
        for index, bucket_key in self.bucket_assignments.items():
            buckets[bucket_key].append(index)
        batch_specs: List[Tuple[List[int], str, float]] = []
        for bucket_key, indices in buckets.items():
            rng.shuffle(indices)
            while len(indices) >= self.batch_size:
                batch = indices[: self.batch_size]
                indices = indices[self.batch_size :]
                weight = sum(self.sample_weights.get(i, 1.0) for i in batch)
                batch_specs.append((batch, bucket_key, max(weight, 1e-6)))
            if indices and self.allow_partial and not self.drop_last:
                batch = indices[:]
                weight = sum(self.sample_weights.get(i, 1.0) for i in batch)
                batch_specs.append((batch, bucket_key, max(weight, 1e-6)))
        if not batch_specs:
            return [], []
        ordered: List[Tuple[float, List[int], str]] = []
        for batch, bucket_key, weight in batch_specs:
            priority = math.log(rng.random() + 1e-8) / weight
            ordered.append((priority, batch, bucket_key))
        ordered.sort(key=lambda item: item[0], reverse=True)
        plan: List[List[int]] = [batch for _, batch, _ in ordered]
        metadata: List[Dict[str, object]] = [
            {"bucket_key": bucket_key, "batch_size": len(batch)}
            for _, batch, bucket_key in ordered
        ]
        return plan, metadata


class KD5CacheBuilder:
    def __init__(
        self,
        dataset_cfg: KD5DatasetConfig,
        cache_dir: Path,
        vae,
        text_embedder,
        device: torch.device,
    ):
        self.dataset_cfg = dataset_cfg
        self.cache_dir = cache_dir
        self.vae = vae
        self.text_embedder = text_embedder
        self.device = device
        self.latent_dir = cache_dir / "latents"
        self.text_dir = cache_dir / "text_embeddings"
        ensure_dirs(self.cache_dir, self.latent_dir, self.text_dir)

    @staticmethod
    def _to_vae_input(pixel_values: torch.Tensor) -> torch.Tensor:
        video = pixel_values.unsqueeze(0)
        video = video.permute(0, 2, 1, 3, 4).contiguous()
        return video

    @torch.no_grad()
    def encode_latents(self, pixel_values: torch.Tensor) -> torch.Tensor:
        video = self._to_vae_input(pixel_values).to(self.device)
        latents = self.vae.encode(video).latent_dist.mode()
        scale = getattr(self.vae.config, "scaling_factor", 0.18215)
        latents = latents * scale
        latents = latents.permute(0, 2, 3, 4, 1).contiguous().squeeze(0)
        return latents.cpu().to(dtype=torch.float16)

    @torch.no_grad()
    def encode_text(
        self, captions: List[str], content_type: str
    ) -> Dict[str, torch.Tensor]:
        embeds, cu_seqlens = self.text_embedder.encode(
            captions, type_of_content=content_type
        )
        return {
            "text_embeds": embeds["text_embeds"].to(dtype=torch.float16).cpu(),
            "pooled_embed": embeds["pooled_embed"].to(dtype=torch.float16).cpu(),
            "cu_seqlens": cu_seqlens.to(dtype=torch.int32).cpu(),
        }

    def build(
        self,
        dataset: KD5RawDataset,
        *,
        progress=None,
        overwrite: bool = False,
        build_latents: bool = True,
        build_text: bool = True,
    ) -> None:
        metadata = {"dataset": self.dataset_cfg.name, "samples": []}
        bucket_registry = getattr(dataset, "bucket_registry", {})
        metadata["bucket_registry"] = {
            key: {"width": bucket.width, "height": bucket.height}
            for key, bucket in bucket_registry.items()
        }
        metadata["default_bucket_key"] = getattr(dataset, "default_bucket_key", None)
        metadata["bucket_diagnostics"] = getattr(dataset, "bucket_diagnostics", {})
        metadata["bucket_settings"] = asdict(self.dataset_cfg.bucket_settings)
        iterator = progress.tqdm if progress else tqdm
        for idx in iterator(
            range(len(dataset)), desc=f"Caching {self.dataset_cfg.name}"
        ):
            sample = dataset[idx]
            cache_key = f"{idx:06d}"
            latent_path = self.latent_dir / f"{cache_key}.pt"
            text_path = self.text_dir / f"{cache_key}.pt"
            latents_ready = latent_path.exists()
            text_ready = text_path.exists()
            if build_latents and (overwrite or not latents_ready):
                latents = self.encode_latents(sample["pixel_values"])
                torch.save(latents, latent_path)
                latents_ready = True
            if build_text and (overwrite or not text_ready):
                text = self.encode_text(
                    [sample["caption"] or ""],
                    "video" if sample["type"] == "video" else "image",
                )
                torch.save(text, text_path)
                text_ready = True
            metadata["samples"].append(
                {
                    "cache_key": cache_key,
                    "caption": sample["caption"],
                    "frames": sample["frames"],
                    "dataset": self.dataset_cfg.name,
                    "latents_ready": latents_ready,
                    "text_ready": text_ready,
                    "bucket": sample.get("bucket"),
                    "bucket_key": sample.get("bucket_key"),
                }
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        with open(
            self.cache_dir / "cache_metadata.json", "w", encoding="utf-8"
        ) as handle:
            json.dump(metadata, handle, indent=2)
