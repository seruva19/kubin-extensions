from dataclasses import dataclass

import gradio as gr

from train_modules.train_tools import text_tip

from ..ui_support import OPTIMIZER_CHOICES, SCHEDULER_CHOICES


@dataclass
class TrainingSection:
    training_container: gr.Accordion
    optim_container: gr.Accordion
    output_container: gr.Accordion
    output_dir: gr.Textbox
    logging_dir: gr.Textbox
    checkpoint_dir: gr.Textbox
    use_lora: gr.Checkbox
    lora_rank: gr.Number
    lora_alpha: gr.Number
    lora_dropout: gr.Number
    lora_targets: gr.Textbox
    optimizer: gr.Dropdown
    scheduler: gr.Dropdown
    learning_rate: gr.Number
    max_grad_norm: gr.Number
    batch_size: gr.Number
    grad_steps: gr.Number
    epochs: gr.Number
    warmup_steps: gr.Number
    max_steps: gr.Textbox
    save_steps: gr.Number
    seed: gr.Number
    precision: gr.Dropdown
    gradient_checkpointing: gr.Checkbox
    resume_from_checkpoint: gr.Textbox
    enable_xformers: gr.Checkbox
    enable_flash: gr.Checkbox
    offload: gr.Checkbox
    snr_gamma: gr.Textbox
    flow_weighting_scheme: gr.Dropdown
    save_every: gr.Number
    keep_last: gr.Number


def build_training_section(default_config, default_lora_targets) -> TrainingSection:
    with gr.Accordion("Training", open=True) as training_container:
        with gr.Row():
            output_dir = gr.Textbox(
                value=default_config.paths.output_dir,
                label="Output directory",
                info="Main directory for training outputs and final model",
            )
            logging_dir = gr.Textbox(
                value=default_config.paths.logging_dir,
                label="Logging directory",
                info="Directory for TensorBoard logs and training metrics",
            )
            checkpoint_dir = gr.Textbox(
                value=default_config.paths.checkpoint_dir,
                label="Checkpoint directory",
                info="Directory for saving intermediate training checkpoints",
            )

        with gr.Row():
            use_lora = gr.Checkbox(
                value=default_config.training.use_lora,
                label="Enable LoRA",
                info="Train using Low-Rank Adaptation instead of full finetuning",
            )
            lora_rank = gr.Number(
                value=default_config.training.lora_rank,
                label="LoRA rank",
                precision=0,
                info="Rank of LoRA matrices (higher = more capacity but slower)",
            )
            lora_alpha = gr.Number(
                value=default_config.training.lora_alpha,
                label="LoRA alpha",
                precision=0,
                info="Scaling factor for LoRA weights (typically equals rank)",
            )
            lora_dropout = gr.Number(
                value=default_config.training.lora_dropout,
                label="LoRA dropout",
                info="Dropout probability for LoRA layers (0.0 to disable)",
            )

        lora_targets = gr.Textbox(
            value=default_lora_targets,
            label="LoRA target modules",
            info=text_tip("Comma separated substrings matched against linear layers"),
        )

        with gr.Row():
            optimizer = gr.Dropdown(
                choices=[value for _, value in OPTIMIZER_CHOICES],
                value=default_config.training.get("optimizer", "adamw8bit"),
                label="Optimizer",
                info="adamw8bit (bitsandbytes), adamw, adam, adafactor, lion",
            )
            scheduler = gr.Dropdown(
                choices=[value for _, value in SCHEDULER_CHOICES],
                value=default_config.training.get("scheduler", "cosine"),
                label="LR scheduler",
                info="cosine, linear, constant, polynomial",
            )
            learning_rate = gr.Number(
                value=default_config.training.learning_rate,
                label="Learning rate",
                info="Initial learning rate (e.g. 1e-4 for LoRA, 1e-5 for full)",
            )
            max_grad_norm = gr.Number(
                value=default_config.training.max_grad_norm,
                label="Max grad norm",
                info="Gradient clipping threshold to prevent exploding gradients",
            )

        with gr.Row():
            batch_size = gr.Number(
                value=default_config.training.train_batch_size,
                label="Batch size",
                precision=0,
                info="Number of samples per training step (limited by VRAM)",
            )
            grad_steps = gr.Number(
                value=default_config.training.gradient_accumulation_steps,
                label="Grad accumulation",
                precision=0,
                info="Accumulate gradients over N steps (effective batch = batch × N)",
            )
            epochs = gr.Number(
                value=default_config.training.num_train_epochs,
                label="Epochs",
                precision=0,
                info="Number of complete passes through the dataset",
            )
            warmup_steps = gr.Number(
                value=default_config.training.warmup_steps,
                label="Warmup steps",
                precision=0,
                info="Gradually increase LR from 0 to target over N steps",
            )
        with gr.Row():
            max_steps = gr.Textbox(
                value=default_config.training.max_train_steps or "",
                label="Max steps (optional)",
                info="Override epochs and train for exactly N steps",
            )
            save_steps = gr.Number(
                value=default_config.training.save_steps,
                label="Checkpoint every",
                precision=0,
                info="Save intermediate checkpoint every N training steps",
            )
            seed = gr.Number(
                value=default_config.training.seed,
                label="Seed",
                precision=0,
                info="Random seed for reproducibility",
            )
            precision = gr.Dropdown(
                choices=["bf16", "fp16", "fp32"],
                value=default_config.training.mixed_precision,
                label="Mixed precision",
                info="bf16 (recommended), fp16 (faster), fp32 (highest precision)",
            )

        with gr.Row():
            gradient_checkpointing = gr.Checkbox(
                value=default_config.training.gradient_checkpointing,
                label="Gradient checkpointing",
            )
            resume_from_checkpoint = gr.Textbox(
                value=default_config.training.resume_from_checkpoint or "",
                label="Resume from checkpoint",
                placeholder="latest or train/kd5/checkpoints/step-001000",
            )

    training_container.elem_classes = ["kubin-accordion"]

    with gr.Accordion("Optimization", open=True) as optim_container:
        with gr.Row():
            enable_xformers = gr.Checkbox(
                value=default_config.optimization.enable_xformers,
                label="Enable xFormers",
            )
            enable_flash = gr.Checkbox(
                value=default_config.optimization.enable_flash_attention,
                label="Flash Attention",
            )
            offload = gr.Checkbox(
                value=default_config.optimization.offload_to_cpu,
                label="Offload encoders to CPU",
            )

        with gr.Row():
            snr_gamma = gr.Textbox(
                value=(
                    ""
                    if default_config.optimization.snr_gamma is None
                    else str(default_config.optimization.snr_gamma)
                ),
                label="SNR gamma",
                info="Min-SNR loss weighting (5.0 recommended, empty=disabled)",
                placeholder="5.0",
            )
            flow_weighting_scheme = gr.Dropdown(
                choices=["none", "logit_normal", "mode"],
                value=default_config.optimization.get(
                    "flow_weighting_scheme", "logit_normal"
                ),
                label="Flow weighting scheme",
                info="SD3-style loss weighting: logit_normal (best), mode, none",
            )

    optim_container.elem_classes = ["kubin-accordion"]

    with gr.Accordion("Outputs", open=True) as output_container:
        with gr.Row():
            save_every = gr.Number(
                value=default_config.outputs.save_every_n_steps,
                label="Keep checkpoint every",
                precision=0,
                info="Permanently save checkpoint every N steps (0 to disable)",
            )
            keep_last = gr.Number(
                value=default_config.outputs.keep_last_n_checkpoints,
                label="Keep last",
                precision=0,
                info="Number of recent checkpoints to keep (older ones deleted)",
            )

    output_container.elem_classes = ["kubin-accordion"]

    return TrainingSection(
        training_container=training_container,
        optim_container=optim_container,
        output_container=output_container,
        output_dir=output_dir,
        logging_dir=logging_dir,
        checkpoint_dir=checkpoint_dir,
        use_lora=use_lora,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_targets=lora_targets,
        optimizer=optimizer,
        scheduler=scheduler,
        learning_rate=learning_rate,
        max_grad_norm=max_grad_norm,
        batch_size=batch_size,
        grad_steps=grad_steps,
        epochs=epochs,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        save_steps=save_steps,
        seed=seed,
        precision=precision,
        gradient_checkpointing=gradient_checkpointing,
        resume_from_checkpoint=resume_from_checkpoint,
        enable_xformers=enable_xformers,
        enable_flash=enable_flash,
        offload=offload,
        snr_gamma=snr_gamma,
        flow_weighting_scheme=flow_weighting_scheme,
        save_every=save_every,
        keep_last=keep_last,
    )
