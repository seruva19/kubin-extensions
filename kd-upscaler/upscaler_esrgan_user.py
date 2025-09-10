import os
import torch
import cv2
import numpy as np
import math
from pathlib import Path
from typing import Dict, Any, List, Optional
from PIL import Image
import glob

from spandrel import MAIN_REGISTRY, ModelDescriptor, ModelLoader, ImageModelDescriptor
from spandrel_extra_arches import EXTRA_REGISTRY

MAIN_REGISTRY.add(*EXTRA_REGISTRY)

_model_cache = {}
_loaded_models = {}


def clear_custom_upscaler_vram():
    global _loaded_models
    for model_path, model_descriptor in _loaded_models.items():
        del model_descriptor
    _loaded_models.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def unload_custom_model(model_path: str):
    global _loaded_models
    if model_path in _loaded_models:
        del _loaded_models[model_path]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class CustomESRGANUpscaler:
    def __init__(self, models_folder: Optional[str] = None):
        if models_folder:
            self.models_folder = Path(models_folder)
        else:
            models_path = Path("models/upscaler")

            if not models_path.exists():
                possible_paths = [
                    Path("models/upscaler"),
                    Path("../models/upscaler"),
                    Path("../../models/upscaler"),
                    Path("../../../models/upscaler"),
                    Path("custom_models"),
                ]

                for path in possible_paths:
                    if path.exists() or path.parent.exists():
                        models_path = path
                        break
                else:
                    print(f"Creating models folder: {models_path.resolve()}")

            self.models_folder = models_path
        self.models_folder.mkdir(parents=True, exist_ok=True)

    def get_available_models(self) -> List[Dict[str, Any]]:
        model_files = list(
            glob.glob(str(self.models_folder / "*.pth"))
            + glob.glob(str(self.models_folder / "*.pt"))
        )

        models = []
        for model_path in model_files:
            try:
                model_name = Path(model_path).stem
                model_info = self._analyze_model(model_path)
                models.append({"name": model_name, "path": model_path, **model_info})
            except Exception as e:
                print(f"Warning: Could not analyze model {model_path}: {e}")
                models.append(
                    {
                        "name": Path(model_path).stem,
                        "path": model_path,
                        "scale": "unknown",
                        "architecture": "unknown",
                        "input_channels": "unknown",
                        "output_channels": "unknown",
                    }
                )

        return models

    def _analyze_model(self, model_path: str) -> Dict[str, Any]:
        try:
            model_descriptor = ModelLoader("cpu").load_from_file(model_path)

            if isinstance(model_descriptor, ImageModelDescriptor):
                arch_name = str(model_descriptor.architecture.id)

                tiling_name = str(model_descriptor.tiling)
                if hasattr(model_descriptor.tiling, "name"):
                    tiling_name = model_descriptor.tiling.name
                elif hasattr(model_descriptor.tiling, "value"):
                    tiling_name = model_descriptor.tiling.value

                return {
                    "scale": model_descriptor.scale,
                    "architecture": arch_name,
                    "input_channels": model_descriptor.input_channels,
                    "output_channels": model_descriptor.output_channels,
                    "supports_half": model_descriptor.supports_half,
                    "supports_bfloat16": getattr(
                        model_descriptor, "supports_bfloat16", False
                    ),
                    "tiling": tiling_name,
                    "tags": getattr(model_descriptor, "tags", []),
                }
            else:
                arch_name = str(model_descriptor.architecture.id)

                return {
                    "scale": getattr(model_descriptor, "scale", 1),
                    "architecture": arch_name,
                    "input_channels": "varies",
                    "output_channels": "varies",
                    "supports_half": model_descriptor.supports_half,
                    "supports_bfloat16": getattr(
                        model_descriptor, "supports_bfloat16", False
                    ),
                    "tiling": "unknown",
                    "tags": getattr(model_descriptor, "tags", []),
                }

        except Exception as e:
            raise ValueError(f"Unsupported model format: {str(e)}")

    def load_model(self, model_path: str, device: torch.device) -> ImageModelDescriptor:
        global _loaded_models

        cache_key = f"{model_path}_{device}"
        if cache_key in _loaded_models:
            return _loaded_models[cache_key]

        try:
            model_descriptor = ModelLoader(device).load_from_file(model_path)

            if not isinstance(model_descriptor, ImageModelDescriptor):
                raise ValueError("Model is not an image super-resolution model")

            model_descriptor.model.eval()
            for param in model_descriptor.model.parameters():
                param.requires_grad = False

            model_descriptor = model_descriptor.to(device)

            _loaded_models[cache_key] = model_descriptor
            return model_descriptor

        except Exception as e:
            raise RuntimeError(f"Failed to load model {model_path}: {str(e)}")


def _upscale_with_tiling(
    img_np: np.ndarray,
    model_descriptor: ImageModelDescriptor,
    device: torch.device,
    use_fp16: bool,
    tile_size: int = 512,
    tile_pad: int = 16,
) -> np.ndarray:
    h, w, c = img_np.shape
    dtype = torch.float16 if use_fp16 else torch.float32

    if h <= tile_size and w <= tile_size:
        img_tensor = torch.from_numpy(img_np.astype(np.float32) / 255.0)
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(device, dtype)

        with torch.no_grad():
            output_tensor = model_descriptor.model(img_tensor)

        output_np = output_tensor.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
        return (output_np * 255.0).clip(0, 255).astype(np.uint8)

    scale = model_descriptor.scale
    output_h, output_w = h * scale, w * scale
    output = np.zeros((output_h, output_w, c), dtype=np.float32)

    y_tiles = math.ceil(h / tile_size)
    x_tiles = math.ceil(w / tile_size)

    for y in range(y_tiles):
        for x in range(x_tiles):
            y_start = y * tile_size
            y_end = min((y + 1) * tile_size, h)
            x_start = x * tile_size
            x_end = min((x + 1) * tile_size, w)

            y_start_pad = max(0, y_start - tile_pad)
            y_end_pad = min(h, y_end + tile_pad)
            x_start_pad = max(0, x_start - tile_pad)
            x_end_pad = min(w, x_end + tile_pad)

            tile = img_np[y_start_pad:y_end_pad, x_start_pad:x_end_pad]

            tile_tensor = torch.from_numpy(tile.astype(np.float32) / 255.0)
            tile_tensor = tile_tensor.permute(2, 0, 1).unsqueeze(0).to(device, dtype)

            with torch.no_grad():
                upscaled_tile_tensor = model_descriptor.model(tile_tensor)

            upscaled_tile = (
                upscaled_tile_tensor.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
            )

            pad_y_start = (y_start - y_start_pad) * scale
            pad_y_end = pad_y_start + (y_end - y_start) * scale
            pad_x_start = (x_start - x_start_pad) * scale
            pad_x_end = pad_x_start + (x_end - x_start) * scale

            output_y_start = y_start * scale
            output_y_end = y_end * scale
            output_x_start = x_start * scale
            output_x_end = x_end * scale

            output[output_y_start:output_y_end, output_x_start:output_x_end] = (
                upscaled_tile[pad_y_start:pad_y_end, pad_x_start:pad_x_end]
            )

    return (output * 255.0).clip(0, 255).astype(np.uint8)


def upscale_custom_esrgan(
    device: torch.device,
    cache_dir: str,
    input_image: Image.Image,
    model_path: str,
    scale: Optional[int] = None,
    tile: int = 0,
    tile_pad: int = 10,
    pre_pad: int = 0,
    half: bool = True,
    clear_vram: bool = False,
) -> Image.Image:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if input_image is None:
        raise ValueError("Input image cannot be None")

    if isinstance(device, str):
        device = torch.device(device)

    if clear_vram:
        clear_custom_upscaler_vram()

    try:
        upscaler = CustomESRGANUpscaler()
        model_descriptor = upscaler.load_model(model_path, device)

        arch_name = str(model_descriptor.architecture.id)
        print(f"Loaded {arch_name} model with {model_descriptor.scale}x scale")

        final_scale = scale if scale is not None else model_descriptor.scale

        if not isinstance(final_scale, int) or final_scale < 1:
            print(
                f"Warning: Invalid scale {final_scale}, using model scale {model_descriptor.scale}"
            )
            final_scale = model_descriptor.scale

        use_fp16 = half and model_descriptor.supports_half and "cuda" in device.type
        if use_fp16:
            model_descriptor.model.half()
        else:
            model_descriptor.model.float()

        img_np = np.array(input_image.convert("RGB"))

        if tile <= 0:
            h, w = img_np.shape[:2]
            if h * w > 2048 * 2048:  # Large image, use tiling
                tile_size = 512
            else:
                tile_size = max(h, w) + 1  # No tiling needed
        else:
            tile_size = tile

        if tile_size < max(img_np.shape[:2]):
            output_np = _upscale_with_tiling(
                img_np, model_descriptor, device, use_fp16, tile_size, tile_pad
            )
        else:
            img_tensor = torch.from_numpy(img_np.astype(np.float32) / 255.0)
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # HWC -> BCHW
            img_tensor = img_tensor.to(device)

            if use_fp16:
                img_tensor = img_tensor.half()

            with torch.no_grad():
                output_tensor = model_descriptor.model(img_tensor)

            output_tensor = output_tensor.squeeze(0).permute(1, 2, 0)  # BCHW -> HWC
            output_np = (
                (output_tensor.cpu().float().numpy() * 255.0)
                .clip(0, 255)
                .astype(np.uint8)
            )

        if final_scale != model_descriptor.scale:
            h, w = output_np.shape[:2]
            target_h = int(input_image.height * final_scale)
            target_w = int(input_image.width * final_scale)

            if (target_w, target_h) != (w, h):
                output_np = cv2.resize(
                    output_np, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4
                )

        return Image.fromarray(output_np)

    except Exception as e:
        raise RuntimeError(f"Failed to upscale image: {str(e)}")


def get_custom_models_list() -> List[Dict[str, Any]]:
    upscaler = CustomESRGANUpscaler()
    return upscaler.get_available_models()


def get_model_info(model_path: str) -> Dict[str, Any]:
    upscaler = CustomESRGANUpscaler()
    return upscaler._analyze_model(model_path)
