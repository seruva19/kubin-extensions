from real_esrgan.model import RealESRGAN


def upscale_resrgan(device, cache_dir, input_image, scale):
    # implementation taken from https://github.com/ai-forever/Real-ESRGAN
    esrgan = RealESRGAN(device, scale=int(scale))
    esrgan.load_weights(f"{cache_dir}/esrgan/RealESRGAN_x{scale}.pth", download=True)

    image = input_image.convert("RGB")
    upscaled_image = esrgan.predict(image)

    return upscaled_image
