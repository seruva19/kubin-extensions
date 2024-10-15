from PIL import Image
from transformers import LlavaForConditionalGeneration, AutoProcessor
from transformers import BitsAndBytesConfig
import torch
import transformers
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import base64
from io import BytesIO

MODEL_ID = "Ertugrul/Pixtral-12B-Captioner-Relaxed"


def resize_image(image, target_size=1024):
    width, height = image.size

    if width < height:
        new_width = target_size
        new_height = int(height * (new_width / width))
    else:
        new_height = target_size
        new_width = int(width * (new_height / height))

    return image.resize((new_width, new_height), Image.LANCZOS)


def pil_to_base64(pil_image, format="PNG"):
    buffer = BytesIO()
    pil_image.save(buffer, format=format)
    img_str = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/{format.lower()};base64,{img_str}"


class PixtralRelaxedModel:
    def __init__(self):
        self.initialized = False
        self.device = "cpu"
        self.quantization_config = None
        self.processor = None
        self.model = None

    def load_model(self, cache_dir, device, quantization):
        print("loading pixtral12b-relaxed model")
        self.device = device

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

            self.model = LlavaForConditionalGeneration.from_pretrained(
                MODEL_ID,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                quantization_config=q_config,
                cache_dir=cache_dir,
            )
            self.processor = AutoProcessor.from_pretrained(
                MODEL_ID, cache_dir=cache_dir
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
                    {"type": "text", "text": model_prompt},
                    {
                        "type": "image",
                    },
                ],
            }
        ]

        prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )
        image = resize_image(input_image)

        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(
            "cuda"
        )
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                generate_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=384,
                    do_sample=True,
                    temperature=0.3,
                    use_cache=True,
                    top_k=20,
                )
        output_text = self.processor.batch_decode(
            generate_ids[:, inputs.input_ids.shape[1] :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0]
        return output_text
