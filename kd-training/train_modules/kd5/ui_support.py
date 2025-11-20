"""Utility helpers and shared constants for the KD5 Gradio UI."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence

from omegaconf import DictConfig, ListConfig, OmegaConf


VARIANT_CHOICES = [
    "5s_sft",
    "5s_pretrain",
    "5s_distil",
    "5s_nocfg",
    "10s_sft",
    "10s_pretrain",
    "10s_distil",
    "10s_nocfg",
]


OPTIMIZER_CHOICES = [
    ("AdamW 8-bit", "adamw8bit"),
    ("AdamW", "adamw"),
    ("Adam", "adam"),
    ("Adafactor", "adafactor"),
    ("Lion", "lion"),
]


SCHEDULER_CHOICES = [
    ("Cosine", "cosine"),
    ("Linear", "linear"),
    ("Constant", "constant"),
    ("Polynomial", "polynomial"),
]


DATASET_FIELD_LABELS = [
    "Name",
    "Raw Path",
    "Cache Name",
    "Resolution",
    "Frames",
    "FPS",
    "Shuffle",
    "Workers",
    "Weight",
    "Bucket Mode",
    "Bucket Settings",
    "Manual Buckets",
]


DATASET_FIELD_DEFAULTS = [
    "",
    "",
    "",
    512,
    16,
    8,
    True,
    2,
    1.0,
    "auto",
    "",
    "",
]


def config_path_for_variant(variant: str) -> str:
    return (
        f"extensions/kd-training/train_modules/kd5/model_configs/config_{variant}.yaml"
    )


def coerce_table(table):
    if table is None:
        return []
    if isinstance(table, list):
        return table
    if hasattr(table, "to_numpy"):
        data = table.to_numpy().tolist()
    elif hasattr(table, "tolist"):
        data = table.tolist()
    else:
        data = list(table)
    coerced = []
    for row in data:
        if isinstance(row, list):
            coerced.append(row)
        else:
            coerced.append(list(row))
    return coerced


def clean_cell(value):
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return value


def prompts_to_rows(prompts):
    if isinstance(prompts, (DictConfig, ListConfig)):
        prompts = OmegaConf.to_container(prompts, resolve=True)
    rows = []
    prompts = coerce_table(prompts)

    for prompt in prompts:
        if hasattr(prompt, "get"):
            getter = prompt.get
            rows.append(
                [
                    getter("text", ""),
                    getter("negative", ""),
                    getter("time_length", 5.0),
                    getter("seed", 42),
                    getter("width", 768),
                    getter("height", 512),
                    getter("bucket", ""),
                ]
            )
            continue

        if prompt is None:
            prompt_list: Sequence = []
        elif isinstance(prompt, list):
            prompt_list = prompt
        elif isinstance(prompt, str):
            prompt_list = [prompt]
        else:
            prompt_list = list(prompt)

        def _item_or_default(index: int, default):
            if index < len(prompt_list):
                value = prompt_list[index]
                if value not in (None, ""):
                    return value
            return default

        rows.append(
            [
                _item_or_default(0, ""),
                _item_or_default(1, ""),
                _item_or_default(2, 5.0),
                _item_or_default(3, 42),
                _item_or_default(4, 768),
                _item_or_default(5, 512),
                _item_or_default(6, ""),
            ]
        )
    return rows


def parse_preview_rows(rows):
    results = []
    coerced_rows = coerce_table(rows)
    # Skip the first row if it contains headers (for Gradio 3.50.2 compatibility)
    if coerced_rows and coerced_rows[0] and coerced_rows[0][0] == "Prompt":
        coerced_rows = coerced_rows[1:]

    for row in coerced_rows:
        row = [clean_cell(cell) for cell in (row or [])]
        if not row or not row[0]:
            continue
        results.append(
            {
                "text": row[0],
                "negative": row[1] or "",
                "time_length": float(row[2]) if row[2] not in (None, "") else 5.0,
                "seed": int(row[3]) if row[3] not in (None, "") else 42,
                "width": int(row[4]) if row[4] not in (None, "") else 768,
                "height": int(row[5]) if row[5] not in (None, "") else 512,
                "bucket": (
                    row[6] if len(row) > 6 and row[6] not in (None, "") else None
                ),
            }
        )
    return results


def _format_bucket_list(buckets):
    formatted = []
    for bucket in buckets or []:
        if not bucket:
            continue
        width = bucket.get("width") if hasattr(bucket, "get") else None
        height = bucket.get("height") if hasattr(bucket, "get") else None
        try:
            width_int = int(width) if width not in (None, "") else None
            height_int = int(height) if height not in (None, "") else None
        except (TypeError, ValueError):
            continue
        if not width_int or not height_int:
            continue
        formatted.append(f"{width_int}x{height_int}")
    return ", ".join(formatted)


def _dataset_column_values(dataset):
    bucket_cfg = dataset.get("bucket") or dataset.get("bucket_settings") or {}
    mode = str(bucket_cfg.get("mode", "auto"))
    manual = dataset.get("aspect_buckets") or bucket_cfg.get("aspect_buckets", [])
    settings = {
        key: value
        for key, value in bucket_cfg.items()
        if key not in {"mode", "aspect_buckets"}
    }
    return [
        dataset.get("name", ""),
        dataset.get("raw_path", ""),
        dataset.get("cache_name", dataset.get("name", "")),
        dataset.get("resolution", 512),
        dataset.get("frames", 16),
        dataset.get("fps", 8),
        dataset.get("shuffle", True),
        dataset.get("num_workers", 2),
        dataset.get("weight", 1.0),
        mode,
        json.dumps(settings, ensure_ascii=False) if settings else "",
        _format_bucket_list(manual),
    ]


def datasets_to_table(datasets):
    dataset_list = list(datasets or [])
    if not dataset_list:
        dataset_list = [
            {
                "name": "video_dataset",
                "raw_path": "train/kd5/videos",
                "cache_name": "video_dataset",
                "resolution": 512,
                "frames": 16,
                "fps": 24,
                "shuffle": True,
                "num_workers": 2,
                "weight": 1.0,
                "aspect_buckets": [],
                "bucket": {"mode": "auto"},
            },
            {
                "name": "image_dataset",
                "raw_path": "train/kd5/images",
                "cache_name": "image_dataset",
                "resolution": 512,
                "frames": 1,
                "fps": 8,
                "shuffle": True,
                "num_workers": 2,
                "weight": 1.0,
                "aspect_buckets": [],
                "bucket": {"mode": "auto"},
            }
        ]

    columns = [_dataset_column_values(ds) for ds in dataset_list]
    rows = []
    for index, label in enumerate(DATASET_FIELD_LABELS):
        row = [label]
        for column in columns:
            value = (
                column[index] if index < len(column) else DATASET_FIELD_DEFAULTS[index]
            )
            row.append(value)
        rows.append(row)
    return normalize_dataset_table(rows)


def dataset_table_headers(table):
    rows = coerce_table(table)
    if not rows:
        return ["Field", "Dataset 1"]
    name_row = rows[0] if rows else []
    headers = ["Field"]
    for index, value in enumerate(name_row[1:], start=1):
        header_label = str(value).strip() if value not in (None, "") else ""
        headers.append(header_label or f"Dataset {index}")
    if len(headers) == 1:
        headers.append("Dataset 1")
    return headers


def normalize_dataset_table(table):
    rows = coerce_table(table)
    # Skip the first row if it contains headers (for Gradio 3.50.2 compatibility)
    if rows and rows[0] and rows[0][0] == "Field":
        rows = rows[1:]

    normalized = []
    for index, label in enumerate(DATASET_FIELD_LABELS):
        if index < len(rows):
            row = rows[index] or []
            values = list(row[1:])
        else:
            values = []
        normalized.append([label] + values)
    return normalized


def _parse_bucket_cell(cell):
    buckets = []
    if isinstance(cell, str):
        text = cell.strip()
        if text:
            if text.startswith("["):
                try:
                    data = json.loads(text)
                except json.JSONDecodeError:
                    data = []
                for item in data or []:
                    if not hasattr(item, "get"):
                        continue
                    width = _as_positive_int(item.get("width"))
                    height = _as_positive_int(item.get("height"))
                    if width and height:
                        buckets.append({"width": width, "height": height})
            else:
                cleaned = text.replace(";", ",")
                for token in [
                    part.strip() for part in cleaned.split(",") if part.strip()
                ]:
                    if "x" not in token.lower():
                        continue
                    left, right = token.lower().split("x", 1)
                    width = _as_positive_int(left)
                    height = _as_positive_int(right)
                    if width and height:
                        buckets.append({"width": width, "height": height})
    elif isinstance(cell, list):
        for item in cell:
            if not hasattr(item, "get"):
                continue
            width = _as_positive_int(item.get("width"))
            height = _as_positive_int(item.get("height"))
            if width and height:
                buckets.append({"width": width, "height": height})
    return buckets


def _as_positive_int(value):
    try:
        value_int = int(value)
    except (TypeError, ValueError):
        return None
    return value_int if value_int > 0 else None


def parse_dataset_table(table):
    rows = normalize_dataset_table(table)
    if not rows:
        return []

    value_rows = []
    for row in rows:
        data_cells = row[1:] if len(row) > 1 else []
        values = [clean_cell(cell) for cell in data_cells]
        value_rows.append(values)

    num_datasets = max((len(values) for values in value_rows), default=0)
    datasets = []
    for dataset_index in range(num_datasets):
        cells = [
            (
                values[dataset_index]
                if dataset_index < len(values)
                else DATASET_FIELD_DEFAULTS[row_index]
            )
            for row_index, values in enumerate(value_rows)
        ]
        raw_path = cells[1]
        if not raw_path:
            continue
        name = cells[0] or ""
        cache_name = cells[2] or (name or Path(raw_path).name)
        try:
            resolution = int(cells[3]) if cells[3] != "" else 512
        except (TypeError, ValueError):
            resolution = 512
        try:
            frames = int(cells[4]) if cells[4] != "" else 16
        except (TypeError, ValueError):
            frames = 16
        try:
            fps = int(cells[5]) if cells[5] != "" else 8
        except (TypeError, ValueError):
            fps = 8
        shuffle_value = cells[6]
        if isinstance(shuffle_value, bool):
            shuffle = shuffle_value
        else:
            shuffle = str(shuffle_value).strip().lower() not in ("false", "0", "no", "")
        try:
            workers = int(cells[7]) if cells[7] != "" else 2
        except (TypeError, ValueError):
            workers = 2
        try:
            weight = float(cells[8]) if cells[8] != "" else 1.0
        except (TypeError, ValueError):
            weight = 1.0
        bucket_mode_value = cells[9] if len(cells) > 9 else "auto"
        bucket_mode = (bucket_mode_value or "auto").strip().lower()
        settings_cell = cells[10] if len(cells) > 10 else ""
        if isinstance(settings_cell, dict):
            bucket_settings = dict(settings_cell)
        else:
            settings_text = str(settings_cell).strip()
            if settings_text:
                try:
                    bucket_settings = json.loads(settings_text)
                    if not isinstance(bucket_settings, dict):
                        bucket_settings = {}
                except json.JSONDecodeError:
                    bucket_settings = {}
            else:
                bucket_settings = {}
        manual_cell = cells[11] if len(cells) > 11 else ""
        aspect_buckets = _parse_bucket_cell(manual_cell)
        dataset_entry = {
            "name": name or cache_name,
            "raw_path": raw_path,
            "cache_name": cache_name,
            "resolution": resolution,
            "frames": frames,
            "fps": fps,
            "shuffle": shuffle,
            "num_workers": workers,
            "weight": weight,
            "aspect_buckets": aspect_buckets,
            "bucket": {**bucket_settings, "mode": bucket_mode},
        }
        datasets.append(dataset_entry)
    return datasets


def default_dataset_table(config):
    datasets = config.get("datasets") or []
    return datasets_to_table(datasets)


def default_preview_rows(config):
    previews = config.get("previews") or {}
    return prompts_to_rows(previews.get("prompts", [])) or [
        [
            "A cinematic drone shot over a futuristic coastal city",
            "low quality, distorted",
            5.0,
            1234,
            768,
            512,
            "",
        ]
    ]


def fill_lora_targets(config) -> str:
    targets = config.get("training", {}).get(
        "lora_target_modules",
        [
            "to_q",
            "to_k",
            "to_v",
            "to_out",
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "linear",
        ],
    )
    if isinstance(targets, list):
        return ",".join(targets)
    return str(targets)


def add_dataset_to_table(table):
    """Add a new dataset column to the table with default values."""
    rows = normalize_dataset_table(table)
    if not rows:
        return datasets_to_table([])

    # Add a new column with default values to each row
    for index, row in enumerate(rows):
        default_value = DATASET_FIELD_DEFAULTS[index]
        row.append(default_value)

    headers = dataset_table_headers(rows)
    # Include headers as the first row for Gradio 3.50.2
    return [headers] + rows


def remove_last_dataset_from_table(table):
    """Remove the last dataset column from the table."""
    rows = normalize_dataset_table(table)
    if not rows:
        return table

    # Check if there's more than one dataset (column 0 is the field label)
    if len(rows[0]) <= 2:  # Only Field column + 1 dataset
        return table  # Don't remove the last dataset

    # Remove the last column from each row
    for row in rows:
        if len(row) > 1:
            row.pop()

    headers = dataset_table_headers(rows)
    # Include headers as the first row for Gradio 3.50.2
    return [headers] + rows


__all__ = [
    "VARIANT_CHOICES",
    "OPTIMIZER_CHOICES",
    "SCHEDULER_CHOICES",
    "DATASET_FIELD_LABELS",
    "DATASET_FIELD_DEFAULTS",
    "config_path_for_variant",
    "coerce_table",
    "clean_cell",
    "prompts_to_rows",
    "parse_preview_rows",
    "datasets_to_table",
    "dataset_table_headers",
    "normalize_dataset_table",
    "parse_dataset_table",
    "default_dataset_table",
    "default_preview_rows",
    "fill_lora_targets",
    "add_dataset_to_table",
    "remove_last_dataset_from_table",
]
