from PIL import Image
from KandiSuperRes import get_SR_pipeline

sr_pipe = None


def upscale_kdsr(device, cache_dir, input_image, steps, batch_size, seed):
    global sr_pipe

    if sr_pipe is None:
        sr_pipe = get_SR_pipeline(device=device, fp16=True, cache_dir=cache_dir)

    sr_image = sr_pipe(input_image, seed=seed, view_batch_size=batch_size, steps=steps)
    return sr_image
