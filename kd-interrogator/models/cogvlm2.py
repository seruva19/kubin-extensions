"""
The code has been adopted from THUDM/CogVLM2
(https://raw.githubusercontent.com/THUDM/CogVLM2/main/basic_demo/cli_demo.py)
"""

import torch
import torch.amp.autocast_mode
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_PATH = "THUDM/cogvlm2-llama3-chat-19B"


class CogVLM2Model:
    def __init__(self):
        self.initialized = False
        self.quantization = None
        self.device = "cpu"
        self.torch_type = torch.float16
        self.tokenizer = None
        self.model = None

    def load_model(self, cache_dir, device, quantization):
        print("loading cog-vlm2 model")

        self.device = device
        self.quantization = quantization
        self.torch_type = (
            torch.bfloat16
            if device == "cuda" and torch.cuda.get_device_capability()[0] >= 8
            else torch.float16
        )

        if not self.initialized:
            self.tokenizer = AutoTokenizer.from_pretrained(
                MODEL_PATH, trust_remote_code=True, cache_dir=cache_dir
            )

            if self.quantization == "4bit":
                self.model = AutoModelForCausalLM.from_pretrained(
                    MODEL_PATH,
                    torch_dtype=self.torch_type,
                    trust_remote_code=True,
                    cache_dir=cache_dir,
                    quantization_config=BitsAndBytesConfig(load_in_4bit=True),
                    low_cpu_mem_usage=True,
                ).eval()
            elif self.quantization == "8bit":
                self.model = AutoModelForCausalLM.from_pretrained(
                    MODEL_PATH,
                    torch_dtype=self.torch_type,
                    trust_remote_code=True,
                    cache_dir=cache_dir,
                    quantization_config=BitsAndBytesConfig(load_in_8bit=True),
                    low_cpu_mem_usage=True,
                ).eval()
            else:
                self.model = (
                    AutoModelForCausalLM.from_pretrained(
                        MODEL_PATH,
                        torch_dtype=self.torch_type,
                        trust_remote_code=True,
                        cache_dir=cache_dir,
                    )
                    .eval()
                    .to(self.device)
                )
            self.initialized = True

    def get_caption(
        self,
        input_image,
        model_prompt,
    ):
        input_by_model = self.model.build_conversation_input_ids(
            self.tokenizer,
            query=model_prompt,
            history=[],
            images=[input_image],
            template_version="chat",
        )

        inputs = {
            "input_ids": input_by_model["input_ids"].unsqueeze(0).to(self.device),
            "token_type_ids": input_by_model["token_type_ids"]
            .unsqueeze(0)
            .to(self.device),
            "attention_mask": input_by_model["attention_mask"]
            .unsqueeze(0)
            .to(self.device),
            "images": [
                [input_by_model["images"][0].to(self.device).to(self.torch_type)]
            ],
        }
        gen_kwargs = {
            "max_new_tokens": 2048,
            "pad_token_id": 128002,
            "top_k": 1,
        }
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs["input_ids"].shape[1] :]
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
