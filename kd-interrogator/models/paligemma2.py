from transformers import (
    PaliGemmaForConditionalGeneration,
    PaliGemmaProcessor,
)
import torch


MODEL_NAME = "google/paligemma2-3b-ft-docci-448"


class Paligemma2Model:
    def __init__(self):

        self.initialized = False
        self.device = "cpu"
        self.model = None

    def load_model(self, cache_dir, device):
        print("loading paligemma2 model")
        self.device = device

        if not self.initialized:
            self.model = (
                PaliGemmaForConditionalGeneration.from_pretrained(
                    MODEL_NAME,
                    torch_dtype=torch.bfloat16,
                    cache_dir=cache_dir,
                )
                .to(device)
                .eval()
            )
            self.processor = PaliGemmaProcessor.from_pretrained(
                MODEL_NAME, cache_dir=cache_dir
            )
            self.initialized = True

    def get_caption(
        self,
        input_image,
        model_prompt,
    ):
        model_inputs = (
            self.processor(text=model_prompt, images=input_image, return_tensors="pt")
            .to(torch.bfloat16)
            .to(self.device)
        )
        input_len = model_inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(
                **model_inputs, max_new_tokens=1024, do_sample=False
            )
            generation = generation[0][input_len:]
            decoded = self.processor.decode(generation, skip_special_tokens=True)
        return decoded
