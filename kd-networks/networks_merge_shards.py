import os
from typing import Set

import gradio as gr
import torch
import safetensors.torch
from safetensors.torch import save_file


def smart_load(path: str):
    """Load torch or safetensors file safely (to CPU)."""
    if path.endswith(".safetensors"):
        return safetensors.torch.load_file(path)
    return torch.load(path, map_location="cpu")


def _dtype_from_str(name: str):
    # Map user friendly names to torch dtypes where available
    if name in ("float16", "fp16"):
        return torch.float16
    if name in ("float32", "fp32"):
        return torch.float32
    if name in ("float8_e4m3fn", "f8e4m3"):
        return getattr(torch, "float8_e4m3fn", None)
    # fallback
    return torch.float16


def merge_model_shards_ui(
    folder: str = ".",
    model_prefix: str = "",
    dtype_name: str = "float8_e4m3fn",
    keep_patterns_text: str = "",
    output_name: str = "",
):
    """
    UI-exposed wrapper. Returns textual log or error.
    """
    logs = []
    try:
        folder = folder or "."
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Folder not found: {folder}")

        keep_patterns: Set[str] = set(
            p.strip() for p in keep_patterns_text.splitlines() if p.strip()
        ) or {
            "norm",
            "head",
            "bias",
            "time_in",
            "vector_in",
            "patch_embedding",
            "text_embedding",
            "time_",
            "img_emb",
            "modulation",
        }

        dtype = _dtype_from_str(dtype_name)
        if dtype is None:
            logs.append(
                f"⚠️ Requested dtype '{dtype_name}' not available in this torch build; falling back to float16."
            )
            dtype = torch.float16

        # find shard files
        shard_files = sorted(
            [
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if f.startswith(model_prefix)
                and (
                    f.endswith(".safetensors")
                    or f.endswith(".pt")
                    or f.endswith(".pth")
                )
            ]
        )
        if not shard_files:
            raise FileNotFoundError(
                f"No shards found in '{folder}' with prefix '{model_prefix}'"
            )

        logs.append(f"🧩 Found {len(shard_files)} shard(s) to merge.")
        merged_sd = {}

        for path in shard_files:
            logs.append(f"🔹 Loading {os.path.basename(path)}")
            current = smart_load(path)
            if not isinstance(current, dict):
                raise ValueError(f"Unexpected checkpoint format in {path}")

            for k, v in current.items():
                if k not in merged_sd:
                    # convert to requested dtype unless pattern is protected
                    if any(pat in k for pat in keep_patterns):
                        merged_sd[k] = v
                    else:
                        # v might already be a torch tensor or numpy; ensure tensor then convert
                        if not isinstance(v, torch.Tensor):
                            v = torch.tensor(v)
                        try:
                            merged_sd[k] = v.to(dtype)
                        except Exception:
                            # fallback to original if conversion fails
                            merged_sd[k] = v
            # free mem
            del current
            torch.cuda.empty_cache()

        logs.append("\n✅ Merged tensors summary:")
        for k, v in merged_sd.items():
            if isinstance(v, torch.Tensor):
                logs.append(f"  {k:<60} {tuple(v.shape)} {v.dtype}")
        logs.append(f"\nTotal tensors: {len(merged_sd)}")

        model_type = model_prefix.replace("_", "-") if model_prefix else "merged"
        if not output_name:
            dt_name = str(dtype).replace("torch.", "")
            output_name = f"{model_type}_merged_{dt_name}.safetensors"

        output_path = os.path.join(folder, output_name)
        logs.append(f"\n💾 Saving merged model to {output_path} ...")
        save_file(
            merged_sd, output_path, metadata={"format": "pt", "model_type": model_type}
        )
        logs.append("✅ Done.")
        return "\n".join(logs)
    except Exception as e:
        return f"❌ Error: {e}"


def networks_merge_shards_ui(kubin):
    """
    Build a Gradio tab for merging model shards.
    """
    with gr.Column() as merge_block:
        with gr.Row():
            folder_in = gr.Textbox(
                label="Shards folder",
                value=".",
                placeholder="Path to folder with shard files",
            )
            prefix_in = gr.Textbox(
                label="Model prefix",
                value="",
                placeholder="Shard filename prefix",
            )
        with gr.Row():
            dtype_in = gr.Dropdown(
                label="Output dtype",
                value="float8_e4m3fn",
                choices=["float8_e4m3fn", "float16", "float32"],
            )
            output_name_in = gr.Textbox(
                label="Output filename (optional)",
                value="",
                placeholder="Filename for merged safetensors",
            )
        keep_text = gr.Textbox(
            label="Keep patterns (one per line, optional)",
            value="\n".join(
                [
                    "norm",
                    "head",
                    "bias",
                    "time_in",
                    "vector_in",
                    "patch_embedding",
                    "text_embedding",
                    "time_",
                    "img_emb",
                    "modulation",
                ]
            ),
            lines=6,
        )
        run_btn = gr.Button("Merge shards")
        log_out = gr.Textbox(label="Log / Output", interactive=False, lines=12)

        def _run(folder, prefix, dtype_name, keep_text_val, output_name):
            return merge_model_shards_ui(
                folder, prefix, dtype_name, keep_text_val, output_name
            )

        run_btn.click(
            fn=_run,
            inputs=[folder_in, prefix_in, dtype_in, keep_text, output_name_in],
            outputs=log_out,
        )

    merge_block.elem_classes = ["kd-networks-merge-block"]
    return merge_block
