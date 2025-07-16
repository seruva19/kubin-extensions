from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

MODEL_ID = "cyan2k/molmo-7B-O-bnb-4bit"


class Molmo7BModel:
    def __init__(self):
        self.initialized = False
        self.device = "cpu"
        self.quantization_config = None
        self.processor = None
        self.model = None

    def load_model(self, cache_dir, device):
        print("loading molmo-7B-O-bnb-4bit model")
        self.device = device

        if not self.initialized:
            arguments = {
                "device_map": "auto",
                "torch_dtype": "auto",
                "trust_remote_code": True,
                # "force_download": True,
                "cache_dir": cache_dir,
            }

            self.processor = AutoProcessor.from_pretrained(MODEL_ID, **arguments)
            self.model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **arguments)
            self.initialized = True

    def get_caption(
        self,
        input_image,
        model_prompt,
    ):
        inputs = self.processor.process(images=[input_image], text=model_prompt)
        inputs = {k: v.to(self.model.device).unsqueeze(0) for k, v in inputs.items()}

        output = self.model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=1024, stop_strings="<|endoftext|>"),
            tokenizer=self.processor.tokenizer,
        )

        generated_tokens = output[0, inputs["input_ids"].size(1) :]
        generated_text = self.processor.tokenizer.decode(
            generated_tokens, skip_special_tokens=True
        )
        return generated_text
