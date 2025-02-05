from PIL import Image

sr_pipe = None


def upscale_kdsr(kubin, device, cache_dir, input_image, steps, batch_size, seed):
    from KandiSuperRes import get_SR_pipeline

    global sr_pipe

    if sr_pipe is None:
        dir = kubin.env_utils.load_env_value("KANDISR_UPSCALER_DIR", cache_dir)
        sr_pipe = get_SR_pipeline(device=device, fp16=True, cache_dir=dir)

    sr_image = sr_pipe(input_image, seed=seed, view_batch_size=batch_size, steps=steps)
    return sr_image
