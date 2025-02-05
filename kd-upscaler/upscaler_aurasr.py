from PIL import Image

aura_pipe = None


def upscale_aura(device, cache_dir, input_image, steps, batch_size, seed):
    from aura_sr import AuraSR

    global aura_pipe

    if aura_pipe is None:
        aura_pipe = AuraSR.from_pretrained("fal/AuraSR-v2")

    upscaled_image = aura_pipe.upscale_4x_overlapped(input_image)
    return upscaled_image
