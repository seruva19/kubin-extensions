import gradio as gr
from pathlib import Path

from omegaconf import OmegaConf

from train_modules.train_tools import (
    load_config_from_path,
    save_config_to_path,
    text_tip,
)
from .train_kd5 import (
    launch_kd5_training,
    cache_kd5_assets,
)
from .kd5_config import (
    load_default_config,
    default_kd5_config_path,
)

from .ui_sections.caching_section import build_caching_section
from .ui_sections.control_panel import build_control_panel
from .ui_sections.datasets_section import build_dataset_section
from .ui_sections.preview_section import build_preview_section
from .ui_sections.training_section import build_training_section
from .ui_sections.variant_section import build_variant_section

from .ui_support import (
    DATASET_FIELD_LABELS,
    OPTIMIZER_CHOICES,
    SCHEDULER_CHOICES,
    VARIANT_CHOICES,
    add_dataset_to_table,
    config_path_for_variant,
    datasets_to_table,
    dataset_table_headers,
    default_dataset_table as build_default_dataset_table,
    default_preview_rows as build_default_preview_rows,
    fill_lora_targets as build_lora_targets,
    normalize_dataset_table,
    parse_dataset_table,
    parse_preview_rows,
    prompts_to_rows,
    remove_last_dataset_from_table,
)


def train_kd5_ui(kubin, tabs):
    default_config = load_default_config()
    config_state = gr.State(default_config)

    def refresh_config_state(values: dict):
        current = config_state.value
        updated = OmegaConf.create(OmegaConf.to_container(current, resolve=True))
        updated.variant.name = values["variant"]
        updated.variant.config = values["config_path"]
        updated.paths.cache_dir = values["cache_dir"]
        updated.paths.output_dir = values["output_dir"]
        updated.paths.logging_dir = values["logging_dir"]
        updated.paths.checkpoint_dir = values["checkpoint_dir"]
        updated.training.use_lora = values["use_lora"]
        updated.training.lora_rank = int(values["lora_rank"])
        updated.training.lora_alpha = int(values["lora_alpha"])
        updated.training.lora_dropout = float(values["lora_dropout"])
        updated.training.lora_target_modules = [
            s.strip() for s in values["lora_targets"].split(",") if s.strip()
        ]
        updated.training.learning_rate = float(values["learning_rate"])
        updated.training.optimizer = values["optimizer"]
        updated.training.scheduler = values["scheduler"]
        updated.training.train_batch_size = int(values["batch_size"])
        updated.training.gradient_accumulation_steps = int(values["grad_steps"])
        updated.training.num_train_epochs = int(values["epochs"])
        updated.training.max_train_steps = (
            None if values["max_steps"] in ("", None) else int(values["max_steps"])
        )
        updated.training.warmup_steps = int(values["warmup_steps"])
        updated.training.mixed_precision = values["precision"]
        updated.training.gradient_checkpointing = values["gradient_checkpointing"]
        updated.training.resume_from_checkpoint = (
            values["resume_from_checkpoint"] or None
        )
        updated.training.max_grad_norm = float(values["max_grad_norm"])
        updated.training.save_steps = int(values["save_steps"])
        updated.training.seed = int(values["seed"])
        updated.datasets = values["datasets"]
        updated.caching.enabled = values["cache_enabled"]
        updated.caching.cache_latents = values["cache_latents"]
        updated.caching.cache_text = values["cache_text"]
        updated.caching.overwrite = values["cache_overwrite"]
        updated.optimization.enable_xformers = values["enable_xformers"]
        updated.optimization.enable_flash_attention = values["enable_flash"]
        updated.optimization.offload_to_cpu = values["offload"]
        updated.optimization.snr_gamma = (
            None if values["snr_gamma"] in ("", None) else float(values["snr_gamma"])
        )
        updated.optimization.flow_weighting_scheme = values["flow_weighting_scheme"]
        updated.outputs.save_every_n_steps = int(values["save_every"])
        updated.outputs.keep_last_n_checkpoints = int(values["keep_last"])
        updated.previews.enabled = values["preview_enabled"]
        updated.previews.step_interval = (
            None
            if values["preview_step"] in ("", None)
            else int(values["preview_step"])
        )
        updated.previews.epoch_interval = (
            None
            if values["preview_epoch"] in ("", None)
            else int(values["preview_epoch"])
        )
        updated.previews.output_dir = values["preview_dir"] or None
        updated.previews.prompts = parse_preview_rows(values["preview_prompts"])
        config_state.value = updated
        return updated

    def collect_values(*inputs):
        (
            variant,
            config_path,
            cache_dir,
            output_dir,
            logging_dir,
            checkpoint_dir,
            dataset_table_data,
            use_lora,
            lora_rank,
            lora_alpha,
            lora_dropout,
            lora_targets,
            learning_rate,
            optimizer,
            scheduler,
            batch_size,
            grad_steps,
            epochs,
            max_steps,
            warmup_steps,
            precision,
            gradient_checkpointing,
            resume_from_checkpoint,
            max_grad_norm,
            save_steps,
            seed,
            cache_enabled,
            cache_latents,
            cache_text,
            cache_overwrite,
            enable_xformers,
            enable_flash,
            offload,
            snr_gamma,
            flow_weighting_scheme,
            save_every,
            keep_last,
            preview_enabled,
            preview_step,
            preview_epoch,
            preview_dir,
            preview_prompts,
        ) = inputs
        datasets = parse_dataset_table(dataset_table_data)
        values = {
            "variant": variant,
            "config_path": config_path,
            "cache_dir": cache_dir,
            "output_dir": output_dir,
            "logging_dir": logging_dir,
            "checkpoint_dir": checkpoint_dir,
            "datasets": datasets,
            "use_lora": use_lora,
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "lora_targets": lora_targets,
            "learning_rate": learning_rate,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "batch_size": batch_size,
            "grad_steps": grad_steps,
            "epochs": epochs,
            "max_steps": max_steps,
            "warmup_steps": warmup_steps,
            "precision": precision,
            "gradient_checkpointing": gradient_checkpointing,
            "resume_from_checkpoint": resume_from_checkpoint,
            "max_grad_norm": max_grad_norm,
            "save_steps": save_steps,
            "seed": seed,
            "cache_enabled": cache_enabled,
            "cache_latents": cache_latents,
            "cache_text": cache_text,
            "cache_overwrite": cache_overwrite,
            "enable_xformers": enable_xformers,
            "enable_flash": enable_flash,
            "offload": offload,
            "snr_gamma": snr_gamma,
            "flow_weighting_scheme": flow_weighting_scheme,
            "save_every": save_every,
            "keep_last": keep_last,
            "preview_enabled": preview_enabled,
            "preview_step": preview_step,
            "preview_epoch": preview_epoch,
            "preview_dir": preview_dir,
            "preview_prompts": preview_prompts,
        }
        cfg = refresh_config_state(values)
        return cfg

    default_dataset_table = build_default_dataset_table(default_config)
    default_lora_targets = build_lora_targets(default_config)
    default_preview_rows = build_default_preview_rows(default_config)

    with gr.Column() as kd5_block:
        variant_controls = build_variant_section(default_config)
        dataset_section = build_dataset_section(default_dataset_table)
        caching_section = build_caching_section(default_config, default_dataset_table)
        training_section = build_training_section(default_config, default_lora_targets)
        preview_section = build_preview_section(default_config, default_preview_rows)
        control_panel = build_control_panel(Path(default_kd5_config_path))

        variant = variant_controls.variant
        config_path = variant_controls.config_path

        dataset_table = dataset_section.table

        cache_dir = caching_section.cache_dir
        cache_enabled = caching_section.cache_enabled
        cache_latents = caching_section.cache_latents
        cache_text = caching_section.cache_text
        cache_overwrite = caching_section.cache_overwrite
        cache_target = caching_section.cache_target
        cache_status = caching_section.cache_status
        cache_latents_btn = caching_section.cache_latents_btn
        cache_text_btn = caching_section.cache_text_btn
        cache_all_btn = caching_section.cache_all_btn

        output_dir = training_section.output_dir
        logging_dir = training_section.logging_dir
        checkpoint_dir = training_section.checkpoint_dir
        use_lora = training_section.use_lora
        lora_rank = training_section.lora_rank
        lora_alpha = training_section.lora_alpha
        lora_dropout = training_section.lora_dropout
        lora_targets = training_section.lora_targets
        optimizer = training_section.optimizer
        scheduler = training_section.scheduler
        learning_rate = training_section.learning_rate
        max_grad_norm = training_section.max_grad_norm
        batch_size = training_section.batch_size
        grad_steps = training_section.grad_steps
        epochs = training_section.epochs
        warmup_steps = training_section.warmup_steps
        max_steps = training_section.max_steps
        save_steps = training_section.save_steps
        seed = training_section.seed
        precision = training_section.precision
        gradient_checkpointing = training_section.gradient_checkpointing
        resume_from_checkpoint = training_section.resume_from_checkpoint
        enable_xformers = training_section.enable_xformers
        enable_flash = training_section.enable_flash
        offload = training_section.offload
        snr_gamma = training_section.snr_gamma
        flow_weighting_scheme = training_section.flow_weighting_scheme
        save_every = training_section.save_every
        keep_last = training_section.keep_last

        preview_enabled = preview_section.preview_enabled
        preview_step = preview_section.preview_step
        preview_epoch = preview_section.preview_epoch
        preview_dir = preview_section.preview_dir
        preview_prompts = preview_section.preview_prompts
        preview_gallery = preview_section.preview_gallery
        refresh_previews = preview_section.refresh_button
        preview_info = preview_section.preview_info

        status = control_panel.status
        train_btn = control_panel.train_btn
        load_btn = control_panel.load_btn
        save_btn = control_panel.save_btn
        config_path_input = control_panel.config_path_input

        def update_dataset_dropdown(rows):
            normalized_rows = normalize_dataset_table(rows)
            parsed = parse_dataset_table(normalized_rows)
            choices = ["All"] + [item["name"] for item in parsed]
            value = "All" if "All" in choices else (choices[0] if choices else None)
            headers = dataset_table_headers(normalized_rows)
            # Include headers as the first row for Gradio 3.50.2
            dataframe_value = [headers] + normalized_rows
            return (
                gr.Dataframe.update(value=dataframe_value),
                gr.Dropdown.update(choices=choices, value=value),
            )

        dataset_table.change(
            update_dataset_dropdown,
            inputs=dataset_table,
            outputs=[dataset_table, cache_target],
        )

        def add_dataset_handler(table):
            new_table = add_dataset_to_table(table)
            normalized_rows = normalize_dataset_table(new_table)
            parsed = parse_dataset_table(normalized_rows)
            choices = ["All"] + [item["name"] for item in parsed]
            value = "All" if "All" in choices else (choices[0] if choices else None)
            return gr.Dataframe.update(value=new_table), gr.Dropdown.update(choices=choices, value=value)

        def remove_dataset_handler(table):
            new_table = remove_last_dataset_from_table(table)
            normalized_rows = normalize_dataset_table(new_table)
            parsed = parse_dataset_table(normalized_rows)
            choices = ["All"] + [item["name"] for item in parsed]
            value = "All" if "All" in choices else (choices[0] if choices else None)
            return gr.Dataframe.update(value=new_table), gr.Dropdown.update(choices=choices, value=value)

        dataset_section.add_dataset_btn.click(
            add_dataset_handler,
            inputs=dataset_table,
            outputs=[dataset_table, cache_target],
        )

        dataset_section.remove_dataset_btn.click(
            remove_dataset_handler,
            inputs=dataset_table,
            outputs=[dataset_table, cache_target],
        )

        def on_variant_change(name):
            return config_path_for_variant(name)

        variant.change(on_variant_change, inputs=variant, outputs=config_path)

        inputs = [
            variant,
            config_path,
            cache_dir,
            output_dir,
            logging_dir,
            checkpoint_dir,
            dataset_table,
            use_lora,
            lora_rank,
            lora_alpha,
            lora_dropout,
            lora_targets,
            learning_rate,
            optimizer,
            scheduler,
            batch_size,
            grad_steps,
            epochs,
            max_steps,
            warmup_steps,
            precision,
            gradient_checkpointing,
            resume_from_checkpoint,
            max_grad_norm,
            save_steps,
            seed,
            cache_enabled,
            cache_latents,
            cache_text,
            cache_overwrite,
            enable_xformers,
            enable_flash,
            offload,
            snr_gamma,
            flow_weighting_scheme,
            save_every,
            keep_last,
            preview_enabled,
            preview_step,
            preview_epoch,
            preview_dir,
            preview_prompts,
        ]

        def run_cache(
            latents_flag,
            text_flag,
            target,
            *args,
            progress=gr.Progress(track_tqdm=True),
        ):
            cfg = collect_values(*args)
            parsed = (
                OmegaConf.to_container(cfg.datasets, resolve=True)
                if hasattr(cfg, "datasets")
                else []
            )
            if not isinstance(parsed, list):
                parsed = []
            dataset_names = [
                item.get("name", f"dataset_{idx}") for idx, item in enumerate(parsed)
            ]
            if target not in ("All", None):
                targets = [target]
            else:
                targets = dataset_names
            if not targets:
                return "No datasets configured"
            if latents_flag and not cfg.caching.cache_latents:
                return "Enable 'Build latents' toggle before caching latents."
            if text_flag and not cfg.caching.cache_text:
                return "Enable 'Build text embeddings' toggle before caching text."
            if not latents_flag and not text_flag:
                return "Nothing selected to cache."
            cache_kd5_assets(
                kubin,
                cfg,
                build_latents=latents_flag,
                build_text=text_flag,
                target_datasets=targets,
                progress=progress,
            )
            return f"Cached {', '.join(targets)}"

        cache_latents_btn.click(
            fn=lambda target, *args, progress=gr.Progress(track_tqdm=True): run_cache(
                True, False, target, *args, progress=progress
            ),
            inputs=[cache_target] + inputs,
            outputs=cache_status,
            _js="() => kubin.UI.taskStarted('KD5 Cache Latents')",
        ).then(
            fn=lambda: gr.update(),
            inputs=None,
            outputs=None,
            _js="() => kubin.UI.taskFinished('KD5 Cache Latents')",
        )

        cache_text_btn.click(
            fn=lambda target, *args, progress=gr.Progress(track_tqdm=True): run_cache(
                False, True, target, *args, progress=progress
            ),
            inputs=[cache_target] + inputs,
            outputs=cache_status,
            _js="() => kubin.UI.taskStarted('KD5 Cache Text')",
        ).then(
            fn=lambda: gr.update(),
            inputs=None,
            outputs=None,
            _js="() => kubin.UI.taskFinished('KD5 Cache Text')",
        )

        cache_all_btn.click(
            fn=lambda target, *args, progress=gr.Progress(track_tqdm=True): run_cache(
                True, True, target, *args, progress=progress
            ),
            inputs=[cache_target] + inputs,
            outputs=cache_status,
            _js="() => kubin.UI.taskStarted('KD5 Cache All')",
        ).then(
            fn=lambda: gr.update(),
            inputs=None,
            outputs=None,
            _js="() => kubin.UI.taskFinished('KD5 Cache All')",
        )

        def refresh_preview_gallery(*args):
            cfg = collect_values(*args)
            preview_dir_value = cfg.previews.get("output_dir")
            output_root = Path(cfg.paths.output_dir)
            preview_dir = (
                Path(preview_dir_value)
                if preview_dir_value
                else output_root / "previews"
            )
            if not preview_dir.is_absolute():
                preview_dir = Path(kubin.root) / preview_dir
            files = []
            if preview_dir.exists():
                supported = {".mp4", ".webm", ".gif", ".png"}
                for path in sorted(preview_dir.rglob("*")):
                    if path.is_file() and path.suffix.lower() in supported:
                        files.append(
                            (str(path), path.relative_to(preview_dir).as_posix())
                        )
            if not files:
                return gr.update(value=[]), "No previews found yet."
            return (
                gr.update(value=files[-12:]),
                f"Loaded {len(files[-12:])} preview file(s).",
            )

        refresh_previews.click(
            fn=refresh_preview_gallery,
            inputs=inputs,
            outputs=[preview_gallery, preview_info],
        )

        def run_training(*args, progress=gr.Progress(track_tqdm=True)):
            cfg = collect_values(*args)
            launch_kd5_training(kubin, cfg, progress=progress)
            return "Training finished"

        train_btn.click(
            fn=run_training,
            inputs=inputs,
            outputs=status,
            _js="() => kubin.UI.taskStarted('KD5 Training')",
        ).then(
            fn=lambda: gr.update(),
            inputs=None,
            outputs=None,
            _js="() => kubin.UI.taskFinished('KD5 Training')",
        ).then(
            fn=refresh_preview_gallery,
            inputs=inputs,
            outputs=[preview_gallery, preview_info],
        )

        def load_config(path: str):
            cfg = load_config_from_path(path)
            config_state.value = cfg
            variant_conf = (
                cfg.variant.name
                if hasattr(cfg.variant, "name")
                else cfg.variant.get("name")
            )
            preview_section = (
                cfg.get("previews", {}) if hasattr(cfg, "get") else cfg.previews
            )
            datasets = cfg.get("datasets", []) if hasattr(cfg, "get") else cfg.datasets
            dataset_table_value = datasets_to_table(datasets)
            headers = dataset_table_headers(dataset_table_value)
            # Include headers as the first row for Gradio 3.50.2
            dataframe_value = [headers] + dataset_table_value
            return (
                variant_conf,
                cfg.variant.get("config", config_path_for_variant(variant_conf)),
                cfg.paths.cache_dir,
                cfg.paths.output_dir,
                cfg.paths.logging_dir,
                cfg.paths.checkpoint_dir,
                gr.Dataframe.update(value=dataframe_value),
                cfg.training.use_lora,
                cfg.training.lora_rank,
                cfg.training.lora_alpha,
                cfg.training.lora_dropout,
                ",".join(cfg.training.get("lora_target_modules", [])),
                cfg.training.learning_rate,
                cfg.training.get("optimizer", "adamw8bit"),
                cfg.training.get("scheduler", "cosine"),
                cfg.training.train_batch_size,
                cfg.training.gradient_accumulation_steps,
                cfg.training.num_train_epochs,
                cfg.training.max_train_steps or "",
                cfg.training.warmup_steps,
                cfg.training.mixed_precision,
                cfg.training.get("gradient_checkpointing", False),
                cfg.training.get("resume_from_checkpoint") or "",
                cfg.training.max_grad_norm,
                cfg.training.save_steps,
                cfg.training.seed,
                cfg.caching.enabled,
                cfg.caching.cache_latents,
                cfg.caching.cache_text,
                cfg.caching.overwrite,
                cfg.optimization.enable_xformers,
                cfg.optimization.enable_flash_attention,
                cfg.optimization.offload_to_cpu,
                (
                    ""
                    if cfg.optimization.snr_gamma is None
                    else str(cfg.optimization.snr_gamma)
                ),
                cfg.optimization.get("flow_weighting_scheme", "logit_normal"),
                cfg.outputs.save_every_n_steps,
                cfg.outputs.keep_last_n_checkpoints,
                (
                    preview_section.get("enabled", False)
                    if hasattr(preview_section, "get")
                    else preview_section.enabled
                ),
                (
                    preview_section.get("step_interval")
                    if hasattr(preview_section, "get")
                    else preview_section.step_interval
                ),
                (
                    preview_section.get("epoch_interval")
                    if hasattr(preview_section, "get")
                    else preview_section.epoch_interval
                ),
                (
                    preview_section.get("output_dir", "")
                    if hasattr(preview_section, "get")
                    else preview_section.output_dir
                ),
                prompts_to_rows(
                    preview_section.get("prompts", [])
                    if hasattr(preview_section, "get")
                    else preview_section.prompts
                ),
                "Configuration loaded",
            )

        load_btn.click(
            fn=load_config,
            inputs=[config_path_input],
            outputs=[
                variant,
                config_path,
                cache_dir,
                output_dir,
                logging_dir,
                checkpoint_dir,
                dataset_table,
                use_lora,
                lora_rank,
                lora_alpha,
                lora_dropout,
                lora_targets,
                learning_rate,
                optimizer,
                scheduler,
                batch_size,
                grad_steps,
                epochs,
                max_steps,
                warmup_steps,
                precision,
                gradient_checkpointing,
                resume_from_checkpoint,
                max_grad_norm,
                save_steps,
                seed,
                cache_enabled,
                cache_latents,
                cache_text,
                cache_overwrite,
                enable_xformers,
                enable_flash,
                offload,
                snr_gamma,
                flow_weighting_scheme,
                save_every,
                keep_last,
                preview_enabled,
                preview_step,
                preview_epoch,
                preview_dir,
                preview_prompts,
                status,
            ],
        ).then(
            fn=update_dataset_dropdown,
            inputs=dataset_table,
            outputs=[dataset_table, cache_target],
        )

        def save_config_fn(path: str, *args):
            cfg = collect_values(*args)
            save_config_to_path(cfg, path)
            return "Configuration saved"

        save_btn.click(
            fn=save_config_fn,
            inputs=[config_path_input] + inputs,
            outputs=status,
        )

    kd5_block.elem_classes = ["kd-training-block"]
    return kd5_block
