"""Configuration dataclasses and helpers for KD5 training."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from omegaconf import DictConfig, OmegaConf


MODULE_DIR = Path(__file__).resolve().parent
TRAIN_MODULES_ROOT = MODULE_DIR.parent
EXTENSION_ROOT = TRAIN_MODULES_ROOT.parent

default_kd5_config_path = (
    TRAIN_MODULES_ROOT / "train_configs" / "config_kd5.yaml"
)
ORIGINAL_CONFIG_ROOT = MODULE_DIR / "model_configs"


OPTIMIZER_REGISTRY = {
    "adamw8bit": "AdamW8bit",
    "adamw": "AdamW",
    "adam": "Adam",
    "adafactor": "Adafactor",
    "lion": "Lion",
}


SCHEDULER_REGISTRY = {
    "cosine": "cosine",
    "linear": "linear",
    "constant": "constant",
    "polynomial": "polynomial",
}


@dataclass
class KD5Variant:
    name: str
    config_path: Path
    scheduler_scale: float = 5.0


@dataclass
class KD5Paths:
    cache_dir: Path
    output_dir: Path
    logging_dir: Path
    checkpoint_dir: Path


@dataclass
class KD5AspectBucket:
    width: int
    height: int


@dataclass
class KD5BucketSettings:
    mode: str = "auto"
    min_size: int = 256
    max_width: int = 1024
    max_height: int = 768
    divisible: int = 64
    step_size: int = 64
    max_buckets: Optional[int] = None
    max_ar_error: float = 0.05
    max_tokens: Optional[int] = None
    include_base_resolution: bool = True
    fallback: Optional[KD5AspectBucket] = None
    seed: Optional[int] = None


@dataclass
class KD5DatasetConfig:
    name: str
    raw_path: Path
    cache_name: str
    resolution: int = 512
    frames: int = 16
    fps: int = 8
    shuffle: bool = True
    num_workers: int = 2
    weight: float = 1.0
    is_validation: bool = False  # Mark dataset as validation set
    aspect_buckets: List[KD5AspectBucket] = field(default_factory=list)
    bucket_settings: KD5BucketSettings = field(default_factory=KD5BucketSettings)


@dataclass
class KD5TrainingSettings:
    use_lora: bool = True
    lora_rank: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            "to_q",
            "to_k",
            "to_v",
            "to_out",
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "linear",
        ]
    )
    learning_rate: float = 1e-4
    optimizer: str = "adamw8bit"
    scheduler: str = "cosine"
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    num_train_epochs: int = 1
    max_train_steps: Optional[int] = None
    warmup_steps: int = 500
    mixed_precision: str = "bf16"
    gradient_checkpointing: bool = True
    max_grad_norm: float = 1.0
    save_steps: int = 500
    validation_steps: int = 0  # Run validation every N steps (0 = disabled)
    validation_batch_size: int = 1  # Batch size for validation
    report_to: str = "tensorboard"
    seed: int = 42
    resume_from_checkpoint: Optional[str] = None


@dataclass
class KD5CachingSettings:
    enabled: bool = True
    cache_latents: bool = True
    cache_text: bool = True
    overwrite: bool = False


@dataclass
class KD5OptimSettings:
    enable_xformers: bool = True
    enable_flash_attention: bool = True
    offload_to_cpu: bool = True
    snr_gamma: Optional[float] = 5.0  # Min-SNR weighting (SD3-style)
    flow_concat_condition: bool = True
    flow_weighting_scheme: str = "logit_normal"  # Better than "none" for quality
    flow_sampling_logit_mean: float = 0.0
    flow_sampling_logit_std: float = 1.0
    flow_sampling_mode_scale: float = 1.29


@dataclass
class KD5OutputSettings:
    save_every_n_steps: int = 1000
    keep_last_n_checkpoints: int = 3


@dataclass
class KD5PreviewPrompt:
    text: str
    negative: str = ""
    time_length: float = 5.0
    seed: int = 42
    width: int = 768
    height: int = 512
    bucket: Optional[str] = None


@dataclass
class KD5PreviewSettings:
    enabled: bool = False
    step_interval: Optional[int] = None
    epoch_interval: Optional[int] = None
    prompts: List[KD5PreviewPrompt] = field(default_factory=list)
    output_dir: Optional[Path] = None


@dataclass
class KD5Config:
    variant: KD5Variant
    paths: KD5Paths
    datasets: List[KD5DatasetConfig]
    training: KD5TrainingSettings
    caching: KD5CachingSettings
    optimization: KD5OptimSettings
    outputs: KD5OutputSettings
    previews: KD5PreviewSettings


class KD5ConfigLoader:
    def __init__(self, kubin_root: Path):
        self.kubin_root = kubin_root
        self.extension_root = EXTENSION_ROOT

    def _resolve(self, value: Optional[str], default: str) -> Path:
        base = value if value not in (None, "") else default
        path = Path(base)
        if not path.is_absolute():
            path = (self.kubin_root / path).resolve()
        return path

    def _resolve_variant(self, variant_cfg: Dict) -> KD5Variant:
        name = variant_cfg.get("name", "5s_sft")
        fallback = (ORIGINAL_CONFIG_ROOT / f"config_{name}.yaml").resolve()
        config_value = variant_cfg.get("config")
        if config_value in (None, ""):
            config_path = fallback
        else:
            config_path = self._resolve(config_value, str(fallback))
            if not config_path.exists():
                rel_path = Path(config_value)
                candidate_paths = [
                    (self.extension_root / rel_path).resolve(),
                ]
                if rel_path.parts and rel_path.parts[0] == "extensions":
                    candidate_paths.append(
                        (self.extension_root / Path(*rel_path.parts[2:])).resolve()
                        if len(rel_path.parts) >= 2 and rel_path.parts[1] == "kd-training"
                        else (self.extension_root / rel_path).resolve()
                    )
                if rel_path.parts and rel_path.parts[0] == "kd-training":
                    candidate_paths.append(
                        (self.extension_root / Path(*rel_path.parts[1:])).resolve()
                    )
                for candidate in candidate_paths:
                    if candidate.exists():
                        config_path = candidate
                        break
        if not config_path.exists():
            raise FileNotFoundError(
                f"Kandinsky config not found at {config_path}. Download from official repo."
            )
        return KD5Variant(
            name=name,
            config_path=config_path,
            scheduler_scale=float(variant_cfg.get("scheduler_scale", 5.0)),
        )

    def _resolve_paths(self, paths_cfg: Dict) -> KD5Paths:
        cache_root = self._resolve(paths_cfg.get("cache_dir"), "train/kd5/cache")
        return KD5Paths(
            cache_dir=cache_root,
            output_dir=self._resolve(paths_cfg.get("output_dir"), "train/kd5/output"),
            logging_dir=self._resolve(paths_cfg.get("logging_dir"), "train/kd5/logs"),
            checkpoint_dir=self._resolve(
                paths_cfg.get("checkpoint_dir"), "train/kd5/checkpoints"
            ),
        )

    def _parse_datasets(
        self, dataset_cfg_list: Iterable[Dict], cache_root: Path
    ) -> List[KD5DatasetConfig]:
        datasets: List[KD5DatasetConfig] = []
        cfg_list = list(dataset_cfg_list or [])
        if not cfg_list:
            datasets.append(
                KD5DatasetConfig(
                    name="dataset_0",
                    raw_path=self._resolve(None, "train/kd5/dataset"),
                    cache_name="dataset_0",
                )
            )
            return datasets

        def _parse_bucket_list(raw_list) -> List[KD5AspectBucket]:
            parsed: List[KD5AspectBucket] = []
            if not raw_list:
                return parsed
            for entry in raw_list:
                if not isinstance(entry, dict):
                    continue
                width_val = entry.get("width")
                height_val = entry.get("height")
                try:
                    width = int(width_val) if width_val not in (None, "") else None
                    height = int(height_val) if height_val not in (None, "") else None
                except (TypeError, ValueError):
                    continue
                if not width or not height or width <= 0 or height <= 0:
                    continue
                parsed.append(KD5AspectBucket(width=width, height=height))
            return parsed

        for idx, cfg in enumerate(cfg_list):
            if not cfg:
                continue
            name = cfg.get("name") or f"dataset_{idx}"
            raw_path = self._resolve(cfg.get("raw_path"), "train/kd5/dataset")
            cache_name = cfg.get("cache_name") or name

            bucket_cfg = cfg.get("bucket") or cfg.get("bucket_settings") or {}
            if isinstance(bucket_cfg, str):
                try:
                    bucket_cfg = json.loads(bucket_cfg)
                except (ValueError, TypeError):
                    bucket_cfg = {}

            resolution = int(cfg.get("resolution", 512))
            base_bucket = KD5AspectBucket(width=resolution, height=resolution)

            manual_list = cfg.get("aspect_buckets") or bucket_cfg.get("aspect_buckets")
            manual_buckets = _parse_bucket_list(manual_list)

            bucket_mode = str(bucket_cfg.get("mode", "auto")).strip().lower()
            if manual_buckets and bucket_mode != "auto":
                bucket_mode = "manual"
            if bucket_mode not in {"auto", "manual"}:
                bucket_mode = "auto"

            fallback_cfg = bucket_cfg.get("fallback") or cfg.get("fallback_bucket")
            fallback_bucket = None
            if isinstance(fallback_cfg, dict):
                try:
                    fb_w = int(fallback_cfg.get("width"))
                    fb_h = int(fallback_cfg.get("height"))
                    if fb_w > 0 and fb_h > 0:
                        fallback_bucket = KD5AspectBucket(width=fb_w, height=fb_h)
                except (TypeError, ValueError):
                    fallback_bucket = None
            if fallback_bucket is None:
                fallback_bucket = base_bucket

            manual_buckets = manual_buckets or [base_bucket]

            datasets.append(
                KD5DatasetConfig(
                    name=name,
                    raw_path=raw_path,
                    cache_name=cache_name,
                    resolution=resolution,
                    frames=int(cfg.get("frames", 16)),
                    fps=int(cfg.get("fps", 8)),
                    shuffle=bool(cfg.get("shuffle", True)),
                    num_workers=int(cfg.get("num_workers", 2)),
                    weight=float(cfg.get("weight", 1.0)),
                    is_validation=bool(cfg.get("is_validation", False)),
                    aspect_buckets=manual_buckets,
                    bucket_settings=KD5BucketSettings(
                        mode=bucket_mode,
                        min_size=int(bucket_cfg.get("min_size", 256)),
                        max_width=int(bucket_cfg.get("max_width", 1024)),
                        max_height=int(bucket_cfg.get("max_height", 768)),
                        divisible=int(bucket_cfg.get("divisible", 64)),
                        step_size=int(bucket_cfg.get("step_size", 64)),
                        max_buckets=(
                            int(bucket_cfg.get("max_buckets"))
                            if bucket_cfg.get("max_buckets") not in (None, "")
                            else None
                        ),
                        max_ar_error=float(bucket_cfg.get("max_ar_error", 0.05)),
                        max_tokens=(
                            int(bucket_cfg.get("max_tokens"))
                            if bucket_cfg.get("max_tokens") not in (None, "")
                            else None
                        ),
                        include_base_resolution=bool(
                            bucket_cfg.get("include_base_resolution", True)
                        ),
                        fallback=fallback_bucket,
                        seed=(
                            int(bucket_cfg.get("seed"))
                            if bucket_cfg.get("seed") not in (None, "")
                            else None
                        ),
                    ),
                )
            )

        return datasets

    def _parse_training(self, training_cfg: Dict) -> KD5TrainingSettings:
        try:
            rank = int(training_cfg.get("lora_rank", 64))
        except (TypeError, ValueError):
            rank = 64
        return KD5TrainingSettings(
            use_lora=bool(training_cfg.get("use_lora", True)),
            lora_rank=rank,
            lora_alpha=int(training_cfg.get("lora_alpha", rank)),
            lora_dropout=float(training_cfg.get("lora_dropout", 0.1)),
            lora_target_modules=list(training_cfg.get("lora_target_modules", []))
            or KD5TrainingSettings().lora_target_modules,
            learning_rate=float(training_cfg.get("learning_rate", 1e-4)),
            optimizer=training_cfg.get("optimizer", "adamw8bit"),
            scheduler=training_cfg.get("scheduler", "cosine"),
            train_batch_size=int(training_cfg.get("train_batch_size", 1)),
            gradient_accumulation_steps=int(
                training_cfg.get("gradient_accumulation_steps", 16)
            ),
            num_train_epochs=int(training_cfg.get("num_train_epochs", 1)),
            max_train_steps=(
                int(training_cfg.get("max_train_steps"))
                if training_cfg.get("max_train_steps") not in (None, "")
                else None
            ),
            warmup_steps=int(training_cfg.get("warmup_steps", 500)),
            mixed_precision=training_cfg.get("mixed_precision", "bf16"),
            gradient_checkpointing=bool(
                training_cfg.get("gradient_checkpointing", True)
            ),
            max_grad_norm=float(training_cfg.get("max_grad_norm", 1.0)),
            save_steps=int(training_cfg.get("save_steps", 500)),
            validation_steps=int(training_cfg.get("validation_steps", 0)),
            report_to=training_cfg.get("report_to", "tensorboard"),
            seed=int(training_cfg.get("seed", 42)),
            resume_from_checkpoint=training_cfg.get("resume_from_checkpoint"),
        )

    def _parse_caching(self, caching_cfg: Dict) -> KD5CachingSettings:
        return KD5CachingSettings(
            enabled=bool(caching_cfg.get("enabled", True)),
            cache_latents=bool(caching_cfg.get("cache_latents", True)),
            cache_text=bool(caching_cfg.get("cache_text", True)),
            overwrite=bool(caching_cfg.get("overwrite", False)),
        )

    def _parse_optim(self, optim_cfg: Dict) -> KD5OptimSettings:
        return KD5OptimSettings(
            enable_xformers=bool(optim_cfg.get("enable_xformers", True)),
            enable_flash_attention=bool(optim_cfg.get("enable_flash_attention", True)),
            offload_to_cpu=bool(optim_cfg.get("offload_to_cpu", True)),
            snr_gamma=(
                float(optim_cfg.get("snr_gamma"))
                if optim_cfg.get("snr_gamma") not in (None, "")
                else None
            ),
            flow_concat_condition=bool(optim_cfg.get("flow_concat_condition", True)),
            flow_weighting_scheme=optim_cfg.get(
                "flow_weighting_scheme", "logit_normal"
            ),
            flow_sampling_logit_mean=float(
                optim_cfg.get("flow_sampling_logit_mean", 0.0)
            ),
            flow_sampling_logit_std=float(
                optim_cfg.get("flow_sampling_logit_std", 1.0)
            ),
            flow_sampling_mode_scale=float(
                optim_cfg.get("flow_sampling_mode_scale", 1.29)
            ),
        )

    def _parse_outputs(self, outputs_cfg: Dict) -> KD5OutputSettings:
        return KD5OutputSettings(
            save_every_n_steps=int(outputs_cfg.get("save_every_n_steps", 1000)),
            keep_last_n_checkpoints=int(outputs_cfg.get("keep_last_n_checkpoints", 3)),
        )

    def _parse_preview_settings(self, previews_cfg: Dict) -> KD5PreviewSettings:
        cfg = previews_cfg or {}
        enabled = bool(cfg.get("enabled", False))
        step_interval = cfg.get("step_interval")
        epoch_interval = cfg.get("epoch_interval")
        output_dir_value = cfg.get("output_dir")
        output_dir = None
        if output_dir_value not in (None, ""):
            output_dir = self._resolve(output_dir_value, output_dir_value)

        def _as_int(value, default):
            if value in (None, ""):
                return default
            try:
                return int(value)
            except (TypeError, ValueError):
                return default

        def _as_float(value, default):
            if value in (None, ""):
                return default
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        prompts: List[KD5PreviewPrompt] = []
        for prompt_cfg in cfg.get("prompts", []) or []:
            if not prompt_cfg:
                continue
            if not isinstance(prompt_cfg, dict):
                continue
            text_value = prompt_cfg.get("text")
            if not text_value:
                continue
            negative = prompt_cfg.get("negative", "")
            time_length = _as_float(prompt_cfg.get("time_length"), 5.0)
            seed = _as_int(prompt_cfg.get("seed"), 42)
            width = _as_int(prompt_cfg.get("width"), 768)
            height = _as_int(prompt_cfg.get("height"), 512)
            bucket_key = prompt_cfg.get("bucket")
            if bucket_key in ("", None):
                bucket_key = None
            prompts.append(
                KD5PreviewPrompt(
                    text=str(text_value),
                    negative=str(negative) if negative is not None else "",
                    time_length=time_length,
                    seed=seed,
                    width=width,
                    height=height,
                    bucket=str(bucket_key) if bucket_key is not None else None,
                )
            )

        return KD5PreviewSettings(
            enabled=enabled,
            step_interval=(
                _as_int(step_interval, None)
                if step_interval not in (None, "")
                else None
            ),
            epoch_interval=(
                _as_int(epoch_interval, None)
                if epoch_interval not in (None, "")
                else None
            ),
            prompts=prompts,
            output_dir=output_dir,
        )

    def load(self, raw: DictConfig) -> KD5Config:
        variant = self._resolve_variant(raw.get("variant", {}))
        paths = self._resolve_paths(raw.get("paths", {}))
        datasets = self._parse_datasets(raw.get("datasets", []), paths.cache_dir)
        training = self._parse_training(raw.get("training", {}))
        caching = self._parse_caching(raw.get("caching", {}))
        optimization = self._parse_optim(raw.get("optimization", {}))
        outputs = self._parse_outputs(raw.get("outputs", {}))
        previews = self._parse_preview_settings(raw.get("previews", {}))
        return KD5Config(
            variant=variant,
            paths=paths,
            datasets=datasets,
            training=training,
            caching=caching,
            optimization=optimization,
            outputs=outputs,
            previews=previews,
        )


def load_default_config() -> DictConfig:
    return OmegaConf.load(str(default_kd5_config_path))


def save_config(config: DictConfig, path: Path) -> None:
    OmegaConf.save(config, path, resolve=True)


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
