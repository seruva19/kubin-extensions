import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_PATH = "SicariusSicariiStuff/X-Ray_Alpha"


class XRayAlphaModel:
    def __init__(self):
        self.initialized = False
        self.device = "cpu"
        self.processor = None
        self.model = None

    def load_model(self, cache_dir, device, quantization="None"):
        print("Loading X-Ray Alpha model...")
        self.device = device

        if not self.initialized:
            torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32

            load_kwargs = {
                "torch_dtype": torch_dtype,
                "device_map": "auto" if device == "cuda" else None,
                "cache_dir": cache_dir,
                "trust_remote_code": True,
                "attn_implementation": "eager",
            }

            if quantization == "8bit" and device == "cuda":
                try:
                    load_kwargs["load_in_8bit"] = True
                except ImportError:
                    print("Warning: bitsandbytes not available for 8-bit quantization")
            elif quantization == "4bit" and device == "cuda":
                try:
                    load_kwargs["load_in_4bit"] = True
                    load_kwargs["bnb_4bit_compute_dtype"] = torch.float16
                    load_kwargs["bnb_4bit_use_double_quant"] = True
                    load_kwargs["bnb_4bit_quant_type"] = "nf4"
                except ImportError:
                    print("Warning: bitsandbytes not available for 4-bit quantization")

            self.model = AutoModelForImageTextToText.from_pretrained(
                MODEL_PATH, **load_kwargs
            )

            if device == "cpu":
                self.model = self.model.to(device)

            self.processor = AutoProcessor.from_pretrained(
                MODEL_PATH,
                cache_dir=cache_dir,
                trust_remote_code=True,
            )

            self.initialized = True
            print("X-Ray Alpha model loaded successfully")

    def get_caption(self, input_image, model_prompt):
        if not self.initialized:
            raise RuntimeError("Model not initialized. Call load_model() first.")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": input_image},
                    {"type": "text", "text": model_prompt},
                ],
            }
        ]

        prompt = self.processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=prompt, images=input_image, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
                num_beams=2,
            )

        output_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        generated_text = output_text[len(prompt) :].strip()

        import re

        cleaned_text = re.sub(r"\[Image #\d+\]", "", generated_text)
        cleaned_text = re.sub(r"^(assistant|Assistant):\s*", "", cleaned_text)

        return cleaned_text.strip()
