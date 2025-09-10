import random
import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from accelerate.utils import set_seed
import sys
from pathlib import Path

current_dir = Path(__file__).parent
hypir_path = current_dir / "HYPIR"
if hypir_path.exists():
    sys.path.insert(0, str(hypir_path))

hypir_model = None
to_tensor = transforms.ToTensor()


def upscale_hypir(
    device, cache_dir, input_image, scale, prompt="", seed=-1, max_size=(8192, 8192)
):
    global hypir_model

    if hypir_model is None:
        try:
            from HYPIR.enhancer.sd2 import SD2Enhancer

            weight_path = os.path.join(cache_dir, "hypir", "HYPIR_sd2.pth")

            if not os.path.exists(weight_path):
                print("HYPIR model weights not found. Downloading...")
                try:
                    current_dir = Path(__file__).parent
                    download_script = current_dir / "download_hypir_weights.py"
                    if download_script.exists():
                        sys.path.insert(0, str(current_dir))
                        from download_hypir_weights import download_hypir_weights

                        download_hypir_weights(cache_dir)
                    else:
                        raise ImportError("Download script not found")
                except ImportError:
                    import urllib.request

                    os.makedirs(os.path.dirname(weight_path), exist_ok=True)
                    model_url = (
                        "https://huggingface.co/lxq007/HYPIR/resolve/main/HYPIR_sd2.pth"
                    )
                    print(f"Downloading HYPIR weights from {model_url}...")
                    urllib.request.urlretrieve(model_url, weight_path)
                    print(f"Downloaded HYPIR weights to {weight_path}")

            hypir_model = SD2Enhancer(
                base_model_path="stabilityai/stable-diffusion-2-1-base",
                weight_path=weight_path,
                lora_modules=[
                    "to_k",
                    "to_q",
                    "to_v",
                    "to_out.0",
                    "conv",
                    "conv1",
                    "conv2",
                    "conv_shortcut",
                    "conv_out",
                    "proj_in",
                    "proj_out",
                    "ff.net.2",
                    "ff.net.0.proj",
                ],
                lora_rank=256,
                model_t=200,
                coeff_t=200,
                device=device,
            )
            hypir_model.init_models()

        except Exception as e:
            raise RuntimeError(f"Failed to initialize HYPIR model: {e}")

    if seed == -1:
        seed = random.randint(0, 2**32 - 1)
    set_seed(seed)

    image = input_image.convert("RGB")

    if max_size is not None:
        out_w, out_h = tuple(int(x * scale) for x in image.size)
        if out_w * out_h > max_size[0] * max_size[1]:
            raise ValueError(
                f"Requested resolution ({out_h}, {out_w}) exceeds maximum pixel limit "
                f"of {max_size[0]} x {max_size[1]} = {max_size[0] * max_size[1]} pixels"
            )

    image_tensor = to_tensor(image).unsqueeze(0)

    try:
        upscaled_image = hypir_model.enhance(
            lq=image_tensor,
            prompt=prompt,
            upscale=scale,
            return_type="pil",
        )[0]

        return upscaled_image

    except Exception as e:
        raise RuntimeError(f"HYPIR upscaling failed: {e}")


def clear_hypir_model():
    global hypir_model
    if hypir_model is not None:
        del hypir_model
        hypir_model = None
        torch.cuda.empty_cache()
