import numpy as np
import torch
import torch.amp.autocast_mode
import torch
from transformers import AutoModel, AutoTokenizer
from PIL import Image
from torchvision import transforms

MODEL_PATH = "internlm/internlm-xcomposer2-4khd-7b"


def pad_to_multiple(image, multiple=336):
    w, h = image.size
    new_w = ((w - 1) // multiple + 1) * multiple
    new_h = ((h - 1) // multiple + 1) * multiple
    result = Image.new(image.mode, (new_w, new_h), (0, 0, 0))
    result.paste(image, (0, 0))
    return result


transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class InternLM2Model:
    def __init__(self):
        self.initialized = False
        self.device = "cpu"
        self.tokenizer = None
        self.model = None

    def load_model(self, cache_dir, device):
        print("loading intern-lm2 model")
        self.device = device

        if not self.initialized:
            self.model = (
                AutoModel.from_pretrained(
                    "internlm/internlm-xcomposer2-4khd-7b",
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    cache_dir=cache_dir,
                )
                .half()
                .cuda()
                .eval()
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                "internlm/internlm-xcomposer2-4khd-7b",
                trust_remote_code=True,
                cache_dir=cache_dir,
            )

    def get_caption(
        self,
        input_image,
        model_prompt,
    ):

        padded_image = pad_to_multiple(input_image)
        input_tensor = transform(padded_image).unsqueeze(0)

        with torch.amp.autocast("cuda"):
            response, _ = self.model.chat(
                self.tokenizer,
                query=model_prompt,
                image=input_tensor,
                hd_num=55,
                history=[],
                do_sample=False,
                num_beams=3,
            )
        return response
