import torch
from PIL import Image
from transformers import AutoModelForCausalLM
from transformers import BitsAndBytesConfig

MODEL_PATH = "AIDC-AI/Ovis2-16B"


class Ovis_2Model:
    def __init__(self):
        self.initialized = False
        self.quantization = None
        self.device = "cpu"
        self.text_tokenizer = None
        self.visual_tokenizer = None
        self.model = None

    def load_model(self, cache_dir, device, quantization):
        print("loading ovis-16b model")

        self.device = device
        self.quantization = quantization

        if not self.initialized:
            q_config = None
            if quantization == "4bit":
                q_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                )

            if quantization == "8bit":
                q_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )

            model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                cache_dir=cache_dir,
                torch_dtype=torch.bfloat16,
                multimodal_max_length=32768,
                trust_remote_code=True,
                quantization_config=q_config,
            )

            if q_config is None:
                model.to(self.device).eval()
            self.text_tokenizer = model.get_text_tokenizer()
            self.visual_tokenizer = model.get_visual_tokenizer()

            self.model = model
            self.initialized = True

    def get_caption(
        self,
        input_image,
        model_prompt,
    ):
        images = [input_image]
        max_partition = 9
        query = f"<image>\n{model_prompt}"

        prompt, input_ids, pixel_values = self.model.preprocess_inputs(
            query, images, max_partition=max_partition
        )
        attention_mask = torch.ne(input_ids, self.text_tokenizer.pad_token_id)
        input_ids = input_ids.unsqueeze(0).to(device=self.model.device)
        attention_mask = attention_mask.unsqueeze(0).to(device=self.model.device)
        if pixel_values is not None:
            pixel_values = pixel_values.to(
                dtype=self.visual_tokenizer.dtype, device=self.visual_tokenizer.device
            )
        pixel_values = [pixel_values]

        with torch.inference_mode():
            gen_kwargs = dict(
                max_new_tokens=1024,
                do_sample=False,
                top_p=None,
                top_k=None,
                temperature=None,
                repetition_penalty=None,
                eos_token_id=self.model.generation_config.eos_token_id,
                pad_token_id=self.text_tokenizer.pad_token_id,
                use_cache=True,
            )
            output_ids = self.model.generate(
                input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                **gen_kwargs,
            )[0]
            output = self.text_tokenizer.decode(output_ids, skip_special_tokens=True)
            return output
