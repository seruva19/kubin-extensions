import torch
import torch.amp.autocast_mode
import torch
import transformers
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import base64
from io import BytesIO

MODEL_PATH = "Qwen/Qwen2-VL-7B-Instruct"


def pil_to_base64(pil_image, format="PNG"):
    buffer = BytesIO()
    pil_image.save(buffer, format=format)
    img_str = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/{format.lower()};base64,{img_str}"


class Qwen2VLModel:
    def __init__(self):
        self.initialized = False
        self.device = "cpu"
        self.processor = None
        self.model = None

    def load_model(self, cache_dir, device):
        print("loading qwen2-vl model")
        self.device = device

        if not self.initialized:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2-VL-7B-Instruct",
                torch_dtype="auto",
                device_map="auto",
                cache_dir=cache_dir,
            )
            self.processor = AutoProcessor.from_pretrained(
                "Qwen/Qwen2-VL-7B-Instruct", cache_dir=cache_dir
            )
            self.initialized = True

    def get_caption(
        self,
        input_image,
        model_prompt,
    ):
        image = pil_to_base64(input_image)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": model_prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        generated_ids = self.model.generate(**inputs, max_new_tokens=1024)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text[0]
