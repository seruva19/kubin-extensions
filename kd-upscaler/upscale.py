from PIL import Image
import numpy as np
import os
import torch

from upscaler_default import upscale_resrgan
from upscaler_kandisr import upscale_kdsr


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
):
    if clear_before_upscale:
        kubin.model.flush()

    if upscaler == "Default (Real-ESRGAN)":
        upscaled_image = upscale_resrgan(device, cache_dir, input_image, scale)
        upscaled_image_path = kubin.fs_utils.save_output(
            os.path.join(output_dir, "upscale"), [upscaled_image]
        )

        return upscaled_image_path

    elif upscaler == "KandiSuperRes":
        upscaled_image = upscale_kdsr(
            device, cache_dir, input_image, steps, batch_size, seed
        )
        upscaled_image_path = kubin.fs_utils.save_output(
            os.path.join(output_dir, "upscale"), [upscaled_image]
        )

        return upscaled_image_path

    elif upscaler == "ESRGAN":
        return []

    else:
        kubin.log(f"upscale method {upscaler} not implemented")
        return []
