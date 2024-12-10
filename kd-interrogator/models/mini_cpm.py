import torch
import torch.amp.autocast_mode
import torch
from transformers import AutoModel, AutoTokenizer


MODEL_PATH = "openbmb/MiniCPM-V-2_6"


class MiniCPMModel:
    def __init__(self):
        self.initialized = False
        self.device = "cpu"
        self.tokenizer = None
        self.model = None

    def load_model(self, cache_dir, device):
        print("loading mini-cpm model")
        self.device = device

        if not self.initialized:
            self.model = AutoModel.from_pretrained(
                MODEL_PATH,
                trust_remote_code=True,
                attn_implementation="sdpa",
                torch_dtype=torch.bfloat16,
                cache_dir=cache_dir,
            ).to(self.device)

            self.tokenizer = AutoTokenizer.from_pretrained(
                MODEL_PATH,
                trust_remote_code=True,
                cache_dir=cache_dir,
            )
            self.initialized = True

    def get_caption(
        self,
        input_image,
        prompt,
    ):
        messages = [{"role": "user", "content": [input_image, prompt]}]
        caption = self.model.chat(image=None, msgs=messages, tokenizer=self.tokenizer)
        return caption
