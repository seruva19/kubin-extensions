"""Training coordinator for KD5."""

from __future__ import annotations

import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from tqdm.auto import tqdm

try:  # Optional diffusers support for flow-matching style sampling
    from diffusers import FlowMatchEulerDiscreteScheduler  # type: ignore
except Exception:  # pragma: no cover - diffusers not installed or misconfigured
    FlowMatchEulerDiscreteScheduler = None  # type: ignore[assignment]

try:  # Optional diffusers loss/sampling utilities
    from diffusers.training_utils import (  # type: ignore
        compute_density_for_timestep_sampling,
        compute_loss_weighting_for_sd3,
    )
except Exception:  # pragma: no cover - diffusers helpers unavailable
    compute_density_for_timestep_sampling = None  # type: ignore[assignment]
    compute_loss_weighting_for_sd3 = None  # type: ignore[assignment]

from .kd5_config import (
    KD5AspectBucket,
    KD5Config,
    KD5DatasetConfig,
    KD5PreviewSettings,
    OPTIMIZER_REGISTRY,
    SCHEDULER_REGISTRY,
    ensure_dirs,
)
from .kd5_datasets import (
    KD5CacheBuilder,
    KD5CachedDataset,
    KD5RawDataset,
    AutoBucketBatchSampler,
    collate_cached_batch,
    _bucket_key,
)
from models.model_50.utils import get_T2V_pipeline
from models.model_50.generation_utils import get_sparse_params
from models.model_50.t2v_pipeline import Kandinsky5T2VPipeline

try:
    import bitsandbytes as bnb

    BNB_AVAILABLE = True
except ImportError:  # pragma: no cover
    BNB_AVAILABLE = False

try:
    from peft import LoraConfig, get_peft_model

    PEFT_AVAILABLE = True
except ImportError:  # pragma: no cover
    PEFT_AVAILABLE = False

logger = get_logger(__name__)


class Kandinsky5Trainer:
    def __init__(self, kubin, config: KD5Config, progress=None):
        self.kubin = kubin
        self.config = config
        self.dataset_lookup = {cfg.name: cfg for cfg in config.datasets}
        self.bucket_registry: Dict[str, KD5AspectBucket] = {}
        self.bucket_plan_metadata: List[Dict[str, object]] = []
        self.progress = progress
        ensure_dirs(
            config.paths.output_dir,
            config.paths.logging_dir,
            config.paths.checkpoint_dir,
        )
        set_seed(config.training.seed)
        project_config = ProjectConfiguration(
            project_dir=str(config.paths.output_dir),
            logging_dir=str(config.paths.logging_dir),
        )
        log_with = (
            config.training.report_to
            if config.training.report_to and config.training.report_to.lower() != "none"
            else None
        )
        self.accelerator = Accelerator(
            mixed_precision=config.training.mixed_precision,
            gradient_accumulation_steps=config.training.gradient_accumulation_steps,
            log_with=log_with,
            project_config=project_config,
        )
        self.device = self.accelerator.device
        self.logger = logger
        self.pipeline_conf = None
        self.using_cache = False
        self.text_embedder_device = None
        self._resume_step = 0
        self.noise_scheduler = None
        self._scheduler_timesteps = None
        self._scheduler_sigmas = None

    def log(self, message: str):
        if self.accelerator.is_main_process:
            self.logger.info(message)

    def _preview_root(self) -> Path:
        preview_cfg = self.config.previews
        if preview_cfg.output_dir is not None:
            return preview_cfg.output_dir
        return self.config.paths.output_dir / "previews"

    def _text_embedder_device(self) -> torch.device:
        if hasattr(self.text_embedder, "embedder") and hasattr(
            self.text_embedder.embedder, "model"
        ):
            return next(self.text_embedder.embedder.model.parameters()).device
        return torch.device("cpu")

    def _module_device(self, module: nn.Module) -> torch.device:
        try:
            return next(module.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def unwrap_model(self, module: nn.Module) -> nn.Module:
        """Mirror diffusers helper to unwrap Accelerate/dynamo-wrapped modules."""
        unwrapped = self.accelerator.unwrap_model(module)
        if hasattr(unwrapped, "_orig_mod"):
            return unwrapped._orig_mod
        return unwrapped

    def _setup_noise_scheduler(self) -> None:
        """Initialise a FlowMatch-style scheduler if diffusers is available."""
        if FlowMatchEulerDiscreteScheduler is None:
            raise RuntimeError(
                "FlowMatchEulerDiscreteScheduler missing. Install diffusers>=0.29."
            )
        try:
            scheduler = FlowMatchEulerDiscreteScheduler()
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                f"Unable to initialise FlowMatchEulerDiscreteScheduler: {exc}"
            ) from exc
        self.noise_scheduler = scheduler
        # Keep CPU copies to avoid regenerating tensors on every call.
        self._scheduler_timesteps = scheduler.timesteps.detach().clone()
        self._scheduler_sigmas = scheduler.sigmas.detach().clone()

    def _sample_noise_parameters(
        self,
        *,
        sample_count: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample diffusion sigmas and matching (normalized) timesteps."""
        if sample_count <= 0:
            raise ValueError("sample_count must be a positive integer")

        if compute_density_for_timestep_sampling is None:
            raise RuntimeError(
                "diffusers.training_utils.compute_density_for_timestep_sampling not "
                "available. Update diffusers to a version that provides Kandinsky 5 "
                "training helpers."
            )
        if (
            self.noise_scheduler is None
            or self._scheduler_timesteps is None
            or self._scheduler_sigmas is None
        ):
            raise RuntimeError("Noise scheduler not initialised before sampling.")

        sigmas = self._scheduler_sigmas.to(device=device, dtype=dtype)
        timesteps = self._scheduler_timesteps.to(
            device=device, dtype=torch.float32
        )

        try:
            density = compute_density_for_timestep_sampling(
                weighting_scheme=getattr(
                    self.config.optimization, "flow_weighting_scheme", "none"
                ),
                batch_size=sample_count,
                logit_mean=getattr(
                    self.config.optimization, "flow_sampling_logit_mean", 0.0
                ),
                logit_std=getattr(
                    self.config.optimization, "flow_sampling_logit_std", 1.0
                ),
                mode_scale=getattr(
                    self.config.optimization, "flow_sampling_mode_scale", 1.29
                ),
            ).to(device=device, dtype=torch.float32)
        except Exception:
            density = torch.rand(sample_count, device=device, dtype=torch.float32)

        max_index = max(len(sigmas) - 1, 0)
        if max_index > 0:
            step_indices = torch.clamp(
                (density * float(max_index)).long(), 0, max_index
            )
        else:
            step_indices = torch.zeros(sample_count, device=device, dtype=torch.long)

        selected_sigmas = sigmas.index_select(0, step_indices)
        selected_timesteps = timesteps.index_select(0, step_indices)

        t_max = torch.max(timesteps)
        t_min = torch.min(timesteps)
        denom = (t_max - t_min).abs().clamp_min(1e-6)
        normalized_time = (selected_timesteps - t_min) / denom
        normalized_time = normalized_time.clamp_(0.0, 1.0)

        return (
            selected_sigmas,
            normalized_time.to(device=device, dtype=torch.float32),
        )

    def _loss_weighting(
        self, sigma_values: torch.Tensor, latents: torch.Tensor
    ) -> torch.Tensor:
        """Compute flow-matching loss weights when diffusers utilities are available."""
        sample_count = latents.shape[0]
        target_shape = (sample_count,) + (1,) * (latents.ndim - 1)
        ones = torch.ones(
            target_shape, device=latents.device, dtype=latents.dtype
        )
        if compute_loss_weighting_for_sd3 is None:
            raise RuntimeError(
                "diffusers.training_utils.compute_loss_weighting_for_sd3 missing. "
                "Update diffusers to a version that ships SD3 loss weighting helpers."
            )
        sigma_flat = sigma_values.reshape(sample_count).to(
            device=latents.device, dtype=latents.dtype
        )
        try:
            weights = compute_loss_weighting_for_sd3(
                weighting_scheme=getattr(
                    self.config.optimization, "flow_weighting_scheme", "none"
                ),
                sigmas=sigma_flat,
            )
        except Exception:
            return ones
        if weights is None:
            return ones
        weights = weights.to(device=latents.device, dtype=latents.dtype)
        if weights.shape == latents.shape:
            return weights
        if weights.numel() == sample_count:
            weights = weights.view(target_shape)
        elif weights.shape != target_shape:
            try:
                weights = weights.reshape(target_shape)
            except Exception:
                return ones
        return weights

    # component setup omitted for brevity (will include actual code below)

    def setup_components(self):
        self.log("Loading Kandinsky 5 pipeline components...")
        device_map = {
            "dit": self.device,
            "vae": self.device,
            "text_embedder": self.device,
        }
        pipeline = get_T2V_pipeline(
            device_map=device_map,
            conf_path=str(self.config.variant.config_path),
            cache_dir=str(self.kubin.params("general", "cache_dir")),
            offload=False,
            use_flash_attention=self.config.optimization.enable_flash_attention,
            use_torch_compile_dit=False,
            use_torch_compile_vae=False,
        )
        self.dit = pipeline.dit
        self.vae = pipeline.vae
        self.text_embedder = pipeline.text_embedder
        self.pipeline_conf = pipeline.conf
        if self.config.optimization.enable_xformers:
            try:
                self.dit.enable_xformers_memory_efficient_attention()
            except Exception:
                self.log("xFormers attention not available")

        # Enable gradient checkpointing for memory efficiency
        if self.config.training.gradient_checkpointing:
            self._enable_gradient_checkpointing()

        # Optionally skip loading text embedder if using fully cached datasets
        if self.config.caching.enabled and self.config.caching.cache_text:
            all_cached = all(
                (self._dataset_cache_dir(cfg) / "cache_metadata.json").exists()
                for cfg in self.config.datasets
            )
            if all_cached and self.config.optimization.offload_to_cpu:
                self.log(
                    "All datasets cached with text - offloading text embedder to CPU"
                )
                self.text_embedder.to("cpu")
        self.vae.eval().requires_grad_(False)
        if hasattr(self.text_embedder, "eval"):
            self.text_embedder.eval()
        if hasattr(self.text_embedder, "requires_grad_"):
            self.text_embedder.requires_grad_(False)
        self._setup_noise_scheduler()

    def prepare_lora(self):
        if not self.config.training.use_lora:
            # Full model training: ensure all parameters are trainable
            self.log("Full model training: enabling gradients for all DiT parameters")
            for param in self.dit.parameters():
                param.requires_grad = True
            return
        if not PEFT_AVAILABLE:
            raise ImportError(
                "peft is required for LoRA training but was not found. "
                "Install peft>=0.6.0 to continue."
            )
        self.log("Configuring LoRA adapters via PEFT")
        lora_config = LoraConfig(
            r=self.config.training.lora_rank,
            lora_alpha=self.config.training.lora_alpha,
            lora_dropout=self.config.training.lora_dropout,
            target_modules=self.config.training.lora_target_modules,
            bias="none",
        )
        self.dit = get_peft_model(self.dit, lora_config)
        self.log("Applied PEFT LoRA adapters")

    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing on DiT transformer blocks for memory efficiency."""
        from torch.utils.checkpoint import checkpoint

        # Wrap visual transformer blocks with gradient checkpointing
        visual_blocks = self.dit.visual_transformer_blocks

        for block in visual_blocks:
            # Store original forward method
            if not hasattr(block, "_original_forward"):
                block._original_forward = block.forward

            # Create checkpointed forward wrapper
            def make_checkpointed_forward(original_forward):
                def checkpointed_forward(*args, **kwargs):
                    if self.dit.training:
                        # Use gradient checkpointing during training
                        return checkpoint(
                            original_forward, *args, use_reentrant=False, **kwargs
                        )
                    else:
                        # Use regular forward during inference
                        return original_forward(*args, **kwargs)

                return checkpointed_forward

            block.forward = make_checkpointed_forward(block._original_forward)

        self.log(f"Enabled gradient checkpointing on {len(visual_blocks)} DiT blocks")

    def _dataset_cache_dir(self, dataset_cfg: KD5DatasetConfig) -> Path:
        return (self.config.paths.cache_dir / dataset_cfg.cache_name).resolve()

    def maybe_build_cache(
        self,
        dataset_cfg: KD5DatasetConfig,
        dataset: KD5RawDataset,
        *,
        build_latents=True,
        build_text=True,
    ):
        if not self.config.caching.enabled:
            self.log("Caching disabled; skipping cache build")
            return
        cache_dir = self._dataset_cache_dir(dataset_cfg)
        ensure_dirs(cache_dir)
        builder = KD5CacheBuilder(
            dataset_cfg=dataset_cfg,
            cache_dir=cache_dir,
            vae=self.vae,
            text_embedder=self.text_embedder,
            device=self.device,
        )
        actual_latents = build_latents and self.config.caching.cache_latents
        actual_text = build_text and self.config.caching.cache_text
        if not actual_latents and not actual_text:
            self.log("Cache builder called with no targets; skipping")
            return
        builder.build(
            dataset,
            progress=self.progress,
            overwrite=self.config.caching.overwrite,
            build_latents=actual_latents,
            build_text=actual_text,
        )
        self.log(f"Cache build complete for {dataset_cfg.name}")

    def _build_datasets(self):
        train_raw_datasets: List[KD5RawDataset] = []
        train_dataset_objs: List[Dataset] = []
        train_dataset_cfgs: List[KD5DatasetConfig] = []
        val_raw_datasets: List[KD5RawDataset] = []
        val_dataset_objs: List[Dataset] = []
        val_dataset_cfgs: List[KD5DatasetConfig] = []
        train_weights: List[float] = []

        use_full_cache = (
            self.config.caching.enabled
            and self.config.caching.cache_latents
            and self.config.caching.cache_text
        )
        if (
            self.config.caching.enabled
            and not use_full_cache
            and self.accelerator.is_main_process
        ):
            self.log(
                "Partial caching configuration detected; falling back to raw datasets"
            )

        for cfg in self.config.datasets:
            raw_ds = KD5RawDataset(cfg)

            if cfg.is_validation:
                val_raw_datasets.append(raw_ds)
            else:
                train_raw_datasets.append(raw_ds)

            dataset_obj: Dataset
            cache_dir = self._dataset_cache_dir(cfg)
            if self.config.caching.enabled and use_full_cache:
                cache_metadata = cache_dir / "cache_metadata.json"
                with self.accelerator.main_process_first():
                    cache_ready = cache_metadata.exists()
                    if self.config.caching.overwrite or not cache_ready:
                        if self.accelerator.is_main_process:
                            self.log(f"Building initial cache for {cfg.name}")
                        if self.accelerator.is_main_process:
                            self.maybe_build_cache(cfg, raw_ds)
                self.accelerator.wait_for_everyone()
                dataset_obj = KD5CachedDataset(cache_dir, cfg)
            else:
                dataset_obj = raw_ds

            weight_val = max(cfg.weight, 1e-6)
            if cfg.is_validation:
                val_dataset_objs.append(dataset_obj)
                val_dataset_cfgs.append(cfg)
            else:
                train_dataset_objs.append(dataset_obj)
                train_dataset_cfgs.append(cfg)
                train_weights.append(weight_val)

        self.using_cache = bool(use_full_cache)
        collate_fn = collate_cached_batch if self.using_cache else None

        lengths = [len(ds) for ds in train_dataset_objs]
        if not lengths or sum(lengths) == 0:
            raise ValueError(
                "Configured datasets are empty; add samples or adjust dataset paths"
            )

        bucket_registry: Dict[str, KD5AspectBucket] = {}
        bucket_assignments: Dict[int, str] = {}
        sample_weights: Dict[int, float] = {}
        global_offset = 0
        for idx, dataset in enumerate(train_dataset_objs):
            dataset_len = len(dataset)
            if dataset_len == 0:
                continue
            dataset_weight = train_weights[idx] if idx < len(train_weights) else 1.0
            weight_per_sample = dataset_weight / max(dataset_len, 1)
            dataset_bucket_keys = getattr(dataset, "bucket_keys", [])
            dataset_registry = getattr(dataset, "bucket_registry", {})
            for key, bucket in dataset_registry.items():
                if key not in bucket_registry:
                    bucket_registry[key] = bucket
            if not dataset_bucket_keys:
                dataset_cfg = train_dataset_cfgs[idx]
                fallback_key = _bucket_key(
                    dataset_cfg.resolution,
                    dataset_cfg.resolution,
                )
                dataset_bucket_keys = [fallback_key] * dataset_len
                if fallback_key not in bucket_registry:
                    bucket_registry[fallback_key] = KD5AspectBucket(
                        width=dataset_cfg.resolution,
                        height=dataset_cfg.resolution,
                    )
            for local_index, bucket_key in enumerate(dataset_bucket_keys):
                sample_index = global_offset + local_index
                bucket_assignments[sample_index] = bucket_key
                sample_weights[sample_index] = max(weight_per_sample, 1e-6)
            global_offset += dataset_len

        concat = ConcatDataset(train_dataset_objs)
        batch_sampler = AutoBucketBatchSampler(
            bucket_assignments=bucket_assignments,
            sample_weights=sample_weights,
            batch_size=self.config.training.train_batch_size,
            allow_partial=True,
            drop_last=False,
            seed=self.config.training.seed,
        )
        _ = len(batch_sampler)
        train_num_workers = (
            max(cfg.num_workers for cfg in train_dataset_cfgs)
            if train_dataset_cfgs
            else 0
        )
        dataloader = DataLoader(
            concat,
            batch_sampler=batch_sampler,
            num_workers=train_num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )
        self.bucket_registry = bucket_registry
        self.bucket_plan_metadata = batch_sampler.plan_metadata

        # Build validation dataloader if validation datasets exist
        val_dataloader = None
        if val_raw_datasets or val_dataset_objs:
            val_datasets: List[Dataset]
            if self.config.caching.enabled and use_full_cache:
                val_datasets = val_dataset_objs
            else:
                val_datasets = val_raw_datasets  # type: ignore[assignment]

            if val_datasets:
                self.log(
                    f"Building validation dataloader with {len(val_datasets)} dataset(s)"
                )
                val_concat = ConcatDataset(val_datasets)
                val_num_workers = (
                    max(cfg.num_workers for cfg in val_dataset_cfgs)
                    if val_dataset_cfgs
                    else 0
                )
                val_bucket_assignments: Dict[int, str] = {}
                val_sample_weights: Dict[int, float] = {}
                val_global_offset = 0
                for idx, dataset in enumerate(val_datasets):
                    dataset_len = len(dataset)
                    if dataset_len == 0:
                        continue
                    dataset_bucket_keys = list(getattr(dataset, "bucket_keys", []))
                    dataset_registry = getattr(dataset, "bucket_registry", {})
                    for key, bucket in dataset_registry.items():
                        if key not in self.bucket_registry:
                            self.bucket_registry[key] = bucket
                    dataset_cfg = (
                        val_dataset_cfgs[idx] if idx < len(val_dataset_cfgs) else None
                    )
                    fallback_key = None
                    if not dataset_bucket_keys:
                        if dataset_cfg is None:
                            continue
                        fallback_key = _bucket_key(
                            dataset_cfg.resolution, dataset_cfg.resolution
                        )
                        dataset_bucket_keys = [fallback_key] * dataset_len
                        if fallback_key not in self.bucket_registry:
                            self.bucket_registry[fallback_key] = KD5AspectBucket(
                                width=dataset_cfg.resolution,
                                height=dataset_cfg.resolution,
                            )
                    weight_per_sample = 1.0 / max(dataset_len, 1)
                    for local_index in range(dataset_len):
                        if local_index < len(dataset_bucket_keys):
                            bucket_key = dataset_bucket_keys[local_index]
                        else:
                            bucket_key = fallback_key
                        if not bucket_key:
                            continue
                        sample_index = val_global_offset + local_index
                        val_bucket_assignments[sample_index] = bucket_key
                        val_sample_weights[sample_index] = max(weight_per_sample, 1e-6)
                    val_global_offset += dataset_len

                val_batch_sampler = AutoBucketBatchSampler(
                    bucket_assignments=val_bucket_assignments,
                    sample_weights=val_sample_weights,
                    batch_size=self.config.training.validation_batch_size,
                    allow_partial=True,
                    drop_last=False,
                    seed=self.config.training.seed,
                )
                _ = len(val_batch_sampler)

                val_dataloader = DataLoader(
                    val_concat,
                    batch_sampler=val_batch_sampler,
                    num_workers=val_num_workers,
                    pin_memory=True,
                    collate_fn=collate_fn,
                )

        return train_raw_datasets, dataloader, val_dataloader

    def _create_optimizer(self, parameters: List[torch.nn.Parameter]):
        opt_name = self.config.training.optimizer
        lr = self.config.training.learning_rate
        if opt_name == "adamw8bit":
            if not BNB_AVAILABLE:
                raise ImportError("bitsandbytes required for AdamW8bit optimizer")
            return bnb.optim.AdamW8bit(
                parameters, lr=lr, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-8
            )
        if opt_name == "adamw":
            return torch.optim.AdamW(
                parameters, lr=lr, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-8
            )
        if opt_name == "adam":
            return torch.optim.Adam(parameters, lr=lr, betas=(0.9, 0.999), eps=1e-8)
        if opt_name == "adafactor":
            from transformers.optimization import Adafactor

            return Adafactor(
                parameters, lr=lr, scale_parameter=False, relative_step=False
            )
        if opt_name == "lion":
            try:
                from lion_pytorch import Lion
            except ImportError as exc:  # pragma: no cover
                raise ImportError("Install lion-pytorch for Lion optimizer") from exc
            return Lion(parameters, lr=lr)
        raise ValueError(f"Unknown optimizer {opt_name}")

    def _create_scheduler(self, optimizer, total_steps: int):
        name = self.config.training.scheduler
        warmup = self.config.training.warmup_steps
        if name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_steps
            )
        if name == "linear":
            from transformers import get_linear_schedule_with_warmup

            return get_linear_schedule_with_warmup(optimizer, warmup, total_steps)
        if name == "constant":
            from transformers import get_constant_schedule_with_warmup

            return get_constant_schedule_with_warmup(optimizer, warmup)
        if name == "polynomial":
            from transformers import get_polynomial_decay_schedule_with_warmup

            return get_polynomial_decay_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup,
                num_training_steps=total_steps,
                lr_end=0.0,
                power=1.0,
            )
        raise ValueError(f"Unknown scheduler {name}")

    def setup_optimizer(self):
        params = [p for p in self.dit.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError(
                "No trainable parameters found. Ensure use_lora=True for LoRA training "
                "or that all model parameters have requires_grad=True for full model training."
            )
        self.log(f"Optimizer will train {sum(p.numel() for p in params):,} parameters")
        optimizer = self._create_optimizer(params)
        total_steps = self.config.training.max_train_steps
        if total_steps is None:
            raise ValueError("max_train_steps must be resolved before optimizer setup")
        scheduler = self._create_scheduler(optimizer, total_steps)
        self.optimizer = optimizer
        self.lr_scheduler = scheduler

    def prepare_accelerator(self, dataloader: DataLoader):
        self.dit, self.optimizer, dataloader, self.lr_scheduler = (
            self.accelerator.prepare(
                self.dit, self.optimizer, dataloader, self.lr_scheduler
            )
        )
        if self.accelerator.is_main_process:
            self.accelerator.init_trackers(
                project_name="kandinsky5-training",
                config={"variant": self.config.variant.name},
            )
        return dataloader

    @torch.no_grad()
    def encode_pixels(
        self, dataset_cfg: KD5DatasetConfig, pixel_values: torch.Tensor
    ) -> torch.Tensor:
        if self.config.optimization.offload_to_cpu:
            self.vae = self.vae.to(self.accelerator.device)
        video = pixel_values.unsqueeze(0).to(self.accelerator.device)
        video = video.permute(0, 2, 1, 3, 4).contiguous()
        latents = self.vae.encode(video).latent_dist.mode()
        scale = getattr(self.vae.config, "scaling_factor", 0.18215)
        latents = latents * scale
        latents = latents.permute(0, 2, 3, 4, 1).contiguous().squeeze(0)
        if self.config.optimization.offload_to_cpu:
            self.vae = self.vae.cpu()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return latents

    @torch.no_grad()
    def encode_text(
        self, captions: List[str], content_type: str
    ) -> Dict[str, torch.Tensor]:
        if self.text_embedder is None:
            raise RuntimeError(
                "Text embedder not loaded. Ensure caching is properly configured."
            )

        # Temporarily move text embedder to GPU if offloaded
        text_device_before = self._text_embedder_device()
        should_restore = (
            self.config.optimization.offload_to_cpu
            and text_device_before.type != "cuda"
        )

        if should_restore:
            self.text_embedder.to(self.accelerator.device)

        try:
            embeds, cu_seqlens = self.text_embedder.encode(
                captions, type_of_content=content_type
            )
            device = self.accelerator.device
            result = {
                "text_embeds": embeds["text_embeds"].to(device),
                "pooled_embed": embeds["pooled_embed"].to(device),
                "cu_seqlens": cu_seqlens.to(device),
            }
        finally:
            # Restore text embedder to CPU if it was offloaded
            if should_restore:
                self.text_embedder.to("cpu")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        return result

    def build_conditioning(
        self, batch: Dict
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        if self.using_cache:
            latents = batch["latents"].to(self.accelerator.device).float()
            text_embeds = {
                "text_embeds": batch["text_embeds"].to(self.accelerator.device).float(),
                "pooled_embed": batch["pooled_embed"]
                .to(self.accelerator.device)
                .float(),
                "cu_seqlens": batch["cu_seqlens"].to(self.accelerator.device),
            }
            frames = batch["frames"].to(self.accelerator.device)
        else:
            pixel_values = batch["pixel_values"].to(self.accelerator.device)
            dataset_names = batch.get("dataset") or []
            if isinstance(dataset_names, torch.Tensor):
                dataset_names = dataset_names.tolist()
            if isinstance(dataset_names, str):
                dataset_names = [dataset_names]
            latents_list = []
            for idx, sample in enumerate(pixel_values):
                dataset_name = dataset_names[idx] if idx < len(dataset_names) else None
                dataset_cfg = self.dataset_lookup.get(
                    dataset_name, self.config.datasets[0]
                )
                latents_list.append(self.encode_pixels(dataset_cfg, sample))
            latents = torch.stack(latents_list, dim=0)
            text_embeds = self.encode_text(batch["caption"], content_type="video")
            frames = torch.tensor(
                [latent.shape[0] for latent in latents],
                device=self.accelerator.device,
                dtype=torch.int64,
            )
        return latents, text_embeds, frames

    def compute_positions(
        self,
        frames: torch.Tensor,
        height: int,
        width: int,
        cu_seqlens: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        patch = getattr(self.pipeline_conf.model.dit_params, "patch_size", [1, 2, 2])
        frame_count = int(frames[0].item()) if frames.numel() > 0 else 0
        patch_t = patch[0] if len(patch) > 0 else 1
        patch_h = patch[1] if len(patch) > 1 else patch_t
        patch_w = patch[2] if len(patch) > 2 else patch_h
        visual_rope_pos = [
            torch.arange(frame_count, device=self.accelerator.device),
            torch.arange(max(1, height // patch_h), device=self.accelerator.device),
            torch.arange(max(1, width // patch_w), device=self.accelerator.device),
        ]
        if cu_seqlens.ndim == 1:
            token_count = int(cu_seqlens[-1].item())
        else:
            token_count = int(cu_seqlens[0, -1].item())
        token_count = max(token_count, 1)
        text_rope_pos = torch.arange(token_count, device=self.accelerator.device)
        return visual_rope_pos, text_rope_pos

    def train_step(self, batch: Dict) -> torch.Tensor:
        latents, text_embeds, frames = self.build_conditioning(batch)
        batch_size, frame_count, height, width, channels = latents.shape
        spatial_height, spatial_width = height, width
        latents = latents.reshape(batch_size * frame_count, height, width, channels)
        noise = torch.randn_like(latents)
        sample_count = latents.shape[0]
        sigma_values, timestep = self._sample_noise_parameters(
            sample_count=sample_count,
            device=self.accelerator.device,
            dtype=latents.dtype,
        )
        sigma = sigma_values.view(sample_count, *([1] * (latents.ndim - 1)))
        time_scalar = timestep.to(device=self.accelerator.device, dtype=torch.float32)
        time_scalar_model = time_scalar.to(dtype=latents.dtype)
        noisy_latents = (1 - sigma) * latents + sigma * noise
        target = noise - latents
        model_input = noisy_latents
        if getattr(self.config.optimization, "flow_concat_condition", False):
            visual_cond = torch.zeros_like(noisy_latents)
            cond_mask_shape = (*noisy_latents.shape[:-1], 1)
            visual_cond_mask = torch.zeros(
                cond_mask_shape,
                device=noisy_latents.device,
                dtype=noisy_latents.dtype,
            )
            model_input = torch.cat(
                [noisy_latents, visual_cond, visual_cond_mask], dim=-1
            )
        visual_rope_pos, text_rope_pos = self.compute_positions(
            frames,
            spatial_height,
            spatial_width,
            text_embeds["cu_seqlens"],
        )
        # Disable sparse attention during training to avoid NABLA FlexAttention OOM issues
        # NABLA attention with torch.compile can trigger 457GB memory allocation
        sparse_params = (
            None
            if self.dit.training
            else get_sparse_params(
                self.pipeline_conf,
                {
                    "visual": (
                        model_input.detach()
                        if getattr(
                            self.config.optimization, "flow_concat_condition", False
                        )
                        else noisy_latents.detach()
                    )
                },
                device=self.accelerator.device,
            )
        )

        model_pred = self.dit(
            model_input,
            text_embeds["text_embeds"],
            text_embeds["pooled_embed"],
            time_scalar_model * 1000.0,
            visual_rope_pos,
            text_rope_pos,
            scale_factor=tuple(self.pipeline_conf.metrics.scale_factor),
            sparse_params=sparse_params,
        )
        residual = model_pred.float() - target.float()
        weight = self._loss_weighting(sigma_values, latents).float()
        weighted = weight * residual.pow(2)
        per_sample_loss = weighted.reshape(sample_count, -1).mean(dim=1)
        snr_gamma = self.config.optimization.snr_gamma
        if snr_gamma is not None:
            sigma_flat = sigma_values.reshape(sample_count).to(
                device=per_sample_loss.device, dtype=per_sample_loss.dtype
            )
            signal = (1.0 - sigma_flat).abs().clamp_min(1e-6)
            noise = sigma_flat.abs().clamp_min(1e-6)
            snr = (signal / noise) ** 2
            snr = snr.clamp_max(float(snr_gamma))
            per_sample_loss = per_sample_loss * snr
        return per_sample_loss.mean()

    @torch.no_grad()
    def validation_step(self, batch: Dict) -> float:
        """Compute validation loss without backprop."""
        latents, text_embeds, frames = self.build_conditioning(batch)
        batch_size, frame_count, height, width, channels = latents.shape
        spatial_height, spatial_width = height, width
        latents = latents.reshape(batch_size * frame_count, height, width, channels)
        noise = torch.randn_like(latents)
        sample_count = latents.shape[0]
        sigma_values, timestep = self._sample_noise_parameters(
            sample_count=sample_count,
            device=self.accelerator.device,
            dtype=latents.dtype,
        )
        sigma = sigma_values.view(sample_count, *([1] * (latents.ndim - 1)))
        time_scalar = timestep.to(device=self.accelerator.device, dtype=torch.float32)
        time_scalar_model = time_scalar.to(dtype=latents.dtype)
        noisy_latents = (1 - sigma) * latents + sigma * noise
        target = noise - latents
        model_input = noisy_latents
        if getattr(self.config.optimization, "flow_concat_condition", False):
            visual_cond = torch.zeros_like(noisy_latents)
            cond_mask_shape = (*noisy_latents.shape[:-1], 1)
            visual_cond_mask = torch.zeros(
                cond_mask_shape,
                device=noisy_latents.device,
                dtype=noisy_latents.dtype,
            )
            model_input = torch.cat(
                [noisy_latents, visual_cond, visual_cond_mask], dim=-1
            )
        visual_rope_pos, text_rope_pos = self.compute_positions(
            frames,
            spatial_height,
            spatial_width,
            text_embeds["cu_seqlens"],
        )
        # No sparse attention during validation
        sparse_params = None

        model_pred = self.dit(
            model_input,
            text_embeds["text_embeds"],
            text_embeds["pooled_embed"],
            time_scalar_model * 1000.0,
            visual_rope_pos,
            text_rope_pos,
            scale_factor=tuple(self.pipeline_conf.metrics.scale_factor),
            sparse_params=sparse_params,
        )
        residual = model_pred.float() - target.float()
        weight = self._loss_weighting(sigma_values, latents).float()
        weighted = weight * residual.pow(2)
        per_sample_loss = weighted.reshape(sample_count, -1).mean(dim=1)
        snr_gamma = self.config.optimization.snr_gamma
        if snr_gamma is not None:
            sigma_flat = sigma_values.reshape(sample_count).to(
                device=per_sample_loss.device, dtype=per_sample_loss.dtype
            )
            signal = (1.0 - sigma_flat).abs().clamp_min(1e-6)
            noise = sigma_flat.abs().clamp_min(1e-6)
            snr = (signal / noise) ** 2
            snr = snr.clamp_max(float(snr_gamma))
            per_sample_loss = per_sample_loss * snr
        return per_sample_loss.mean().item()

    @torch.no_grad()
    def run_validation(self, val_dataloader: DataLoader, global_step: int):
        """Run full validation pass and log metrics."""
        if val_dataloader is None or len(val_dataloader) == 0:
            return

        if not self.accelerator.is_main_process:
            return

        self.log(f"Running validation at step {global_step}...")

        # Set model to eval mode
        self.dit.eval()

        total_loss = 0.0
        num_batches = 0

        try:
            for batch in val_dataloader:
                loss = self.validation_step(batch)
                if isinstance(loss, torch.Tensor):
                    loss_value = float(loss.detach().cpu().item())
                else:
                    loss_value = float(loss)
                total_loss += loss_value
                num_batches += 1
        except Exception as exc:
            self.log(f"Validation error: {exc}")
            self.dit.train()
            return

        # Compute average validation loss
        avg_val_loss = total_loss / max(num_batches, 1)

        # Log to tracking system
        self.accelerator.log(
            {"val/loss": float(avg_val_loss), "val/num_batches": num_batches},
            step=global_step,
        )

        self.log(f"Validation loss: {avg_val_loss:.4f} ({num_batches} batches)")

        # Restore model to training mode
        self.dit.train()

    def save_checkpoint(self, step: int, epoch: int = 0):
        if not self.accelerator.is_main_process:
            return
        ckpt_dir = self.config.paths.checkpoint_dir / f"step-{step:06d}"
        ensure_dirs(ckpt_dir)
        self.accelerator.save_state(str(ckpt_dir))
        state_path = ckpt_dir / "trainer_state.json"
        try:
            with state_path.open("w", encoding="utf-8") as handle:
                json.dump({"step": step, "epoch": epoch}, handle)
        except Exception as exc:  # pragma: no cover
            self.log(f"Failed to write trainer state file: {exc}")
        checkpoints = sorted(self.config.paths.checkpoint_dir.iterdir())
        surplus = len(checkpoints) - self.config.outputs.keep_last_n_checkpoints
        for path in checkpoints[: max(0, surplus)]:
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)

    def generate_previews(self, current_step: int, current_epoch: int, reason: str):
        forecasts = self.config.previews
        if not forecasts.enabled or not forecasts.prompts:
            return
        if not self.accelerator.is_main_process:
            return
        preview_root = self._preview_root()
        ensure_dirs(preview_root)
        run_dir = preview_root / reason
        ensure_dirs(run_dir)
        device = self.accelerator.device
        dit = self.unwrap_model(self.dit)
        was_training = dit.training
        dit.eval()
        text_device_before = self._text_embedder_device()
        vae_device_before = self._module_device(self.vae)
        moved_text_embedder = False
        moved_vae = False
        pipeline = None
        try:
            if text_device_before != device:
                self.text_embedder = self.text_embedder.to(device)
                moved_text_embedder = True
            if vae_device_before != device:
                self.vae = self.vae.to(device)
                moved_vae = True
            pipeline = Kandinsky5T2VPipeline(
                device_map={"dit": device, "vae": device, "text_embedder": device},
                dit=dit,
                text_embedder=self.text_embedder,
                vae=self.vae,
                conf=self.pipeline_conf,
                offload=False,
            )
            scheduler_scale = self.config.variant.scheduler_scale
            saved = []
            for idx, prompt in enumerate(forecasts.prompts):
                time_length = float(prompt.time_length)
                suffix = "mp4" if time_length > 0 else "png"
                file_path = run_dir / f"{idx:02d}_{prompt.seed}.{suffix}"
                bucket_dims = None
                if getattr(prompt, "bucket", None):
                    bucket = self.bucket_registry.get(prompt.bucket)
                    if bucket is None:
                        self.log(
                            f"Preview prompt requested unknown bucket {prompt.bucket}; falling back to explicit size"
                        )
                    else:
                        bucket_dims = (bucket.width, bucket.height)
                width = int(bucket_dims[0] if bucket_dims else prompt.width)
                height = int(bucket_dims[1] if bucket_dims else prompt.height)
                try:
                    with torch.no_grad():
                        pipeline(
                            text=prompt.text,
                            time_length=time_length,
                            width=width,
                            height=height,
                            seed=int(prompt.seed),
                            scheduler_scale=scheduler_scale,
                            negative_caption=prompt.negative,
                            expand_prompts=False,
                            save_path=str(file_path),
                            progress=False,
                        )
                    saved.append(file_path)
                except Exception as exc:  # pragma: no cover
                    self.log(f"Preview generation failed: {exc}")
                finally:
                    if device.type == "cuda" and torch.cuda.is_available():
                        torch.cuda.empty_cache()
            if saved:
                self.log(
                    f"Saved {len(saved)} preview(s) to {run_dir.relative_to(preview_root)}"
                )
        finally:
            if moved_text_embedder:
                self.text_embedder = self.text_embedder.to(text_device_before)
            if moved_vae:
                self.vae = self.vae.to(vae_device_before)
            if pipeline is not None:
                del pipeline
            if device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
            dit.train(was_training)

    def _resolve_resume_checkpoint(self) -> Optional[Path]:
        resume_value = self.config.training.resume_from_checkpoint
        if not resume_value:
            return None
        resume_value = str(resume_value).strip()
        if not resume_value:
            return None
        if resume_value.lower() == "latest":
            checkpoints = sorted(
                p for p in self.config.paths.checkpoint_dir.iterdir() if p.is_dir()
            )
            if not checkpoints:
                return None
            return checkpoints[-1]
        path = Path(resume_value)
        if not path.is_absolute():
            path = (self.config.paths.checkpoint_dir / path).resolve()
        if path.exists() and path.is_dir():
            return path
        return None

    def _load_resume_state(self) -> Tuple[int, int]:
        checkpoint_path = self._resolve_resume_checkpoint()
        if checkpoint_path is None:
            if self.config.training.resume_from_checkpoint:
                self.log(
                    f"Requested resume checkpoint "
                    f"{self.config.training.resume_from_checkpoint!r} not found; starting fresh"
                )
            return 0, 0
        self.log(f"Resuming training from checkpoint: {checkpoint_path}")
        self.accelerator.load_state(str(checkpoint_path))
        resume_step = 0
        resume_epoch = 0
        state_path = checkpoint_path / "trainer_state.json"
        if state_path.exists():
            try:
                with state_path.open("r", encoding="utf-8") as handle:
                    data = json.load(handle)
                resume_step = int(data.get("step", 0))
                resume_epoch = int(data.get("epoch", 0))
            except Exception:
                resume_step = 0
                resume_epoch = 0
        if resume_step > 0:
            self.log(
                f"Resumed global step set to {resume_step}, epoch set to {resume_epoch}"
            )
        return max(resume_step, 0), max(resume_epoch, 0)

    def train(self):
        self.setup_components()
        self.prepare_lora()
        raw_datasets, dataloader, val_dataloader = self._build_datasets()
        if self.config.training.max_train_steps is None:
            steps_per_epoch = max(1, len(dataloader))
            # Account for gradient accumulation to get actual optimizer update steps
            num_update_steps_per_epoch = math.ceil(
                steps_per_epoch / self.config.training.gradient_accumulation_steps
            )
            total_steps = (
                num_update_steps_per_epoch * self.config.training.num_train_epochs
            )
            self.config.training.max_train_steps = total_steps
        self.setup_optimizer()
        dataloader = self.prepare_accelerator(dataloader)
        # Prepare validation dataloader with accelerator for distributed training compatibility
        if val_dataloader is not None:
            val_dataloader = self.accelerator.prepare(val_dataloader)
        total_steps = self.config.training.max_train_steps

        # Check if validation is enabled
        validation_enabled = (
            val_dataloader is not None and self.config.training.validation_steps > 0
        )
        if validation_enabled:
            self.log(
                f"Validation enabled: running every {self.config.training.validation_steps} steps"
            )

        self.accelerator.wait_for_everyone()
        # Explicitly set model to training mode
        self.dit.train()
        self._resume_step, self._resume_epoch = self._load_resume_state()
        self.accelerator.wait_for_everyone()
        global_step = self._resume_step

        # Skip batches if resuming from checkpoint
        if global_step > 0:
            grad_accum = max(1, self.config.training.gradient_accumulation_steps)
            micro_batches_to_skip = global_step * grad_accum
            self.log(
                f"Skipping first {micro_batches_to_skip} micro-batches "
                f"({global_step} optimizer steps) to resume training"
            )
            dataloader = self.accelerator.skip_first_batches(
                dataloader, micro_batches_to_skip
            )
        previews_cfg = self.config.previews
        step_interval = (
            previews_cfg.step_interval
            if previews_cfg.enabled
            and previews_cfg.step_interval
            and previews_cfg.step_interval > 0
            else None
        )
        epoch_interval = (
            previews_cfg.epoch_interval
            if previews_cfg.enabled
            and previews_cfg.epoch_interval
            and previews_cfg.epoch_interval > 0
            else None
        )
        progress_bar = (
            self.progress.tqdm(range(total_steps), desc="KD5 Training")
            if self.progress
            else tqdm(range(total_steps), desc="KD5 Training")
        )
        if global_step > 0:
            progress_bar.update(min(global_step, total_steps))
        if global_step >= total_steps:
            progress_bar.close()
            if self.accelerator.is_main_process:
                self.log("Reached configured max_train_steps; skipping training loop")
            return
        # Use saved epoch if available, otherwise estimate
        starting_epoch = self._resume_epoch if hasattr(self, "_resume_epoch") else 0
        if global_step > 0 and starting_epoch == 0:
            # Fallback: estimate epoch based on steps completed
            steps_per_epoch = max(1, len(dataloader))
            # Account for gradient accumulation
            updates_per_epoch = math.ceil(
                steps_per_epoch / self.config.training.gradient_accumulation_steps
            )
            starting_epoch = min(
                global_step // updates_per_epoch,
                self.config.training.num_train_epochs - 1,
            )

        last_epoch_completed = starting_epoch - 1
        for epoch in range(starting_epoch, self.config.training.num_train_epochs):
            # Set epoch for proper batch shuffling
            if hasattr(dataloader.sampler, "set_epoch"):
                dataloader.sampler.set_epoch(epoch)
            elif hasattr(dataloader.batch_sampler, "set_epoch"):
                dataloader.batch_sampler.set_epoch(epoch)

            if global_step >= total_steps:
                break
            for batch in dataloader:
                if global_step >= total_steps:
                    break
                with self.accelerator.accumulate(self.dit):
                    loss = self.train_step(batch)
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        torch.nn.utils.clip_grad_norm_(
                            self.dit.parameters(),
                            self.config.training.max_grad_norm,
                        )
                    self.optimizer.step()
                    # Only step scheduler when gradients are actually synced (optimizer update)
                    if self.accelerator.sync_gradients:
                        self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                # Only increment global_step when optimizer actually updates
                if self.accelerator.sync_gradients:
                    global_step += 1

                    # Update progress bar only on optimizer updates
                    if self.accelerator.is_main_process:
                        progress_bar.update(1)
                        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

                    # Log metrics only on optimizer updates
                    if self.accelerator.is_main_process:
                        self.accelerator.log(
                            {"train/loss": float(loss.item())}, step=global_step
                        )

                    # Run validation if enabled and at validation interval
                    if (
                        validation_enabled
                        and global_step % self.config.training.validation_steps == 0
                    ):
                        self.run_validation(val_dataloader, global_step)

                    # Generate previews if at step interval
                    if step_interval and global_step % step_interval == 0:
                        self.generate_previews(
                            current_step=global_step,
                            current_epoch=epoch + 1,
                            reason=f"step-{global_step:06d}",
                        )

                    # Save checkpoint if at save interval
                    if (
                        self.accelerator.is_main_process
                        and self.config.training.save_steps > 0
                        and global_step % self.config.training.save_steps == 0
                    ):
                        self.save_checkpoint(global_step, epoch)

            last_epoch_completed = epoch
            # Generate epoch-level previews after batch loop completes
            if epoch_interval and global_step > 0 and (epoch + 1) % epoch_interval == 0:
                self.generate_previews(
                    current_step=global_step,
                    current_epoch=epoch + 1,
                    reason=f"epoch-{epoch + 1:04d}",
                )
        progress_bar.close()
        if self.accelerator.is_main_process:
            final_epoch = (
                last_epoch_completed if last_epoch_completed >= 0 else starting_epoch
            )
            self.save_checkpoint(global_step, final_epoch)
