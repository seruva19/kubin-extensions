import torch
import torch.amp.autocast_mode
import torch
from transformers import (
    AutoTokenizer,
    AutoModel,
)
from PIL import Image
from torchvision import transforms
from optimum.quanto import QuantizedModelForCausalLM, qint4, qint8

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
        self.quantization_config = None
        self.tokenizer = None
        self.model = None

    def load_model(self, cache_dir, device, quantization):
        print("loading intern-lm2 model")
        self.device = device

        if not self.initialized:
            self.tokenizer = AutoTokenizer.from_pretrained(
                MODEL_PATH,
                trust_remote_code=True,
                cache_dir=cache_dir,
            )

            self.model = AutoModel.from_pretrained(
                MODEL_PATH,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                # quantization_config=self.quantization_config,
                cache_dir=cache_dir,
            )

            # self.set_quantize(self.model, quantization)
            self.model.to(device).eval()

            self.initialized = True

    def set_quantize(self, model, quantization):
        if quantization == "4bit":
            self.model = QuantizedModelForCausalLM.quantize(
                self.model, weights=qint4, exclude="lm_head"
            )

        if quantization == "8bit":
            self.model = QuantizedModelForCausalLM.quantize(
                self.model, weights=qint8, exclude="lm_head"
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
