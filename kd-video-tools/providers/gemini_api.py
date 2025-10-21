import os
import torch
from transformers import AutoModel, AutoTokenizer
from transformers import BitsAndBytesConfig
from decord import VideoReader, cpu
from PIL import Image
from google import genai
from google.genai import types

GEMINI_MODEL_ID = "gemini-2.0-flash"


def peek_gemini_key():
    current_file_path = os.path.abspath(__file__)
    current_directory = os.path.dirname(current_file_path)
    gemini_key_path = os.path.join(current_directory, "gemini.key")
    with open(gemini_key_path, "r") as file:
        gemini_key = file.read()
    return gemini_key


def init_gemini(state):
    state["name"] = GEMINI_MODEL_ID

    def interrogate(video_path, question):
        video_bytes = open(video_path, "rb").read()

        gemini_key = peek_gemini_key()
        client = genai.Client(api_key=gemini_key)
        response = client.models.generate_content(
            model="models/gemini-2.0-flash",
            contents=types.Content(
                parts=[
                    types.Part(text=question),
                    types.Part(
                        inline_data=types.Blob(data=video_bytes, mime_type="video/mp4")
                    ),
                ]
            ),
        )
        print(response.text)
        return response.text

    state["fn"] = interrogate
