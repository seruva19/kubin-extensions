import torch
import torch.amp.autocast_mode
import torch
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor,
    BitsAndBytesConfig,
)
from qwen_vl_utils import process_vision_info
import base64
from io import BytesIO

MODEL_PATH = "Ertugrul/Qwen2-VL-7B-Captioner-Relaxed"


def pil_to_base64(pil_image, format="PNG"):
    buffer = BytesIO()
    pil_image.save(buffer, format=format)
    img_str = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/{format.lower()};base64,{img_str}"


class Qwen2VLRelaxedModel:
    def __init__(self):
        self.initialized = False
        self.device = "cpu"
        self.processor = None
        self.model = None

    def load_model(self, cache_dir, device):
        print("loading qwen2-vl-relaxed model")
        self.device = device

        if not self.initialized:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                MODEL_PATH,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                cache_dir=cache_dir,
            ).to(self.device)

            self.processor = AutoProcessor.from_pretrained(
                MODEL_PATH, cache_dir=cache_dir
            )
            self.initialized = True

    def get_caption(
        self,
        input_image,
        model_prompt,
    ):
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                    },
                    {"type": "text", "text": model_prompt},
                ],
            }
        ]

        text_prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text_prompt], images=[input_image], padding=True, return_tensors="pt"
        )
        inputs = inputs.to(self.device)

        with torch.no_grad():
            with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=True,
                    temperature=0.7,
                    use_cache=True,
                    top_k=50,
                )

        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0]

        return output_text
