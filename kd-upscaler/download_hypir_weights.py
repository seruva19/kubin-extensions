import os
import requests
from pathlib import Path
from tqdm import tqdm


def download_file(url, destination, chunk_size=8192):
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))

    with open(destination, "wb") as file, tqdm(
        desc=f"Downloading {os.path.basename(destination)}",
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                file.write(chunk)
                progress_bar.update(len(chunk))


def download_hypir_weights(cache_dir):
    model_url = "https://huggingface.co/lxq007/HYPIR/resolve/main/HYPIR_sd2.pth"

    hypir_cache_dir = os.path.join(cache_dir, "hypir")
    os.makedirs(hypir_cache_dir, exist_ok=True)

    model_path = os.path.join(hypir_cache_dir, "HYPIR_sd2.pth")

    if os.path.exists(model_path):
        print(f"HYPIR model weights already exist at: {model_path}")
        return model_path

    print(f"Downloading HYPIR model weights to: {model_path}")

    try:
        download_file(model_url, model_path)
        print(f"Successfully downloaded HYPIR model weights to: {model_path}")
        return model_path
    except Exception as e:
        print(f"Error downloading HYPIR model weights: {e}")

        if os.path.exists(model_path):
            os.remove(model_path)
        raise
