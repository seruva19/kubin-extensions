from PIL import Image
import numpy as np
import os
import torch

from upscaler_default import upscale_resrgan
from upscaler_kandisr import upscale_kdsr
from upscaler_aurasr import upscale_aura
from upscaler_esrgan_user import upscale_custom_esrgan, get_custom_models_list
from upscaler_hypir import upscale_hypir


def upscale_with(
    kubin,
    upscaler,
    device,
    cache_dir,
    scale,
    output_dir,
    input_image,
    clear_before_upscale,
    steps,
    batch_size,
    seed,
    model_path=None,
    hypir_prompt="",
):
    if clear_before_upscale:
        kubin.model.flush()

    if model_path and upscaler == "ESRGAN (user)":
        upscaled_image = upscale_custom_esrgan(
            device,
            cache_dir,
            input_image,
            model_path,
            scale=None,
            clear_vram=clear_before_upscale,
        )
        upscaled_image_path = kubin.fs_utils.save_output(
            os.path.join(output_dir, "upscale"), [upscaled_image]
        )
        return upscaled_image_path

    if upscaler == "Default (Real-ESRGAN)":
        upscaled_image = upscale_resrgan(device, cache_dir, input_image, scale)
        upscaled_image_path = kubin.fs_utils.save_output(
            os.path.join(output_dir, "upscale"), [upscaled_image]
        )

        return upscaled_image_path

    elif upscaler == "KandiSuperRes":
        upscaled_image = upscale_kdsr(
            kubin, device, cache_dir, input_image, steps, batch_size, seed
        )
        upscaled_image_path = kubin.fs_utils.save_output(
            os.path.join(output_dir, "upscale"), [upscaled_image]
        )

        return upscaled_image_path

    elif upscaler == "AuraSR-v2":
        upscaled_image = upscale_aura(
            device, cache_dir, input_image, steps, batch_size, seed
        )
        upscaled_image_path = kubin.fs_utils.save_output(
            os.path.join(output_dir, "upscale"), [upscaled_image]
        )

        return upscaled_image_path

    elif upscaler == "HYPIR":
        upscaled_image = upscale_hypir(
            device, cache_dir, input_image, int(scale), hypir_prompt, seed
        )
        upscaled_image_path = kubin.fs_utils.save_output(
            os.path.join(output_dir, "upscale"), [upscaled_image]
        )

        return upscaled_image_path

    else:
        kubin.log(f"upscale method {upscaler} not implemented")
        return []
