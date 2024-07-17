import json
import gradio as gr
from pathlib import Path
import yaml
import uuid
import os
from enum import Enum
from transformers import AutoModelForCausalLM, AutoTokenizer
import requests


class LLM_Type(Enum):
    HF_LOCAL = "HF Repo Id"
    OLLAMA_API = "Ollama API"


title = "LLM Enhancer"
targets = ["t2i", "i2i", "mix", "inpaint", "outpaint"]


def format_prompt(source_prompt, llm_prompt):
    return llm_prompt.format(prompt=source_prompt)


local_model = None
local_tokenizer = None


def enhance_with_local_llm(kubin, model_name, source_prompt, llm_prompt):
    global local_model
    global local_tokenizer

    if local_model is None:
        cache_dir = kubin.params("general", "cache_dir")
        device = kubin.params("general", "device")

        local_model = AutoModelForCausalLM.from_pretrained(
            model_name, cache_dir=cache_dir, device=device
        )
        local_tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=cache_dir, device=device
        )
    inputs = local_tokenizer.encode(
        format_prompt(source_prompt, llm_prompt),
        return_tensors="pt",
        add_special_tokens=False,
    )
    outputs = local_model.generate(inputs, max_length=1000, num_return_sequences=1)
    response = local_tokenizer.decode(outputs[0], skip_special_tokens=True)

    if model_name == "Intel/neural-chat-7b-v3-1":
        response = response.split("### Assistant:")[-1]
    return response


def enhance_with_llama_api(
    kubin, ollama_model, ollama_api_url, source_prompt, llm_prompt
):
    ollama_url = f"{ollama_api_url}/api/generate"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": ollama_model,
        "prompt": format_prompt(source_prompt, llm_prompt),
        "stream": False,
    }

    payload_json = json.dumps(payload)
    response = requests.post(ollama_url, headers=headers, data=payload_json)

    if response.status_code == 200:
        data = response.json()
        llm_response = data["response"]
        return llm_response
    else:
        print(
            f"There was an error connecting Ollama: [{response.status_code}] {response.text}"
        )
    return source_prompt


def enhance(
    kubin,
    target,
    params,
    enabled,
    type,
    hf_model_name,
    ollama_model_name,
    ollama_url,
    llm_prompt,
):
    if enabled:
        if type == LLM_Type.HF_LOCAL.name:
            print(f"Using HF model {hf_model_name}")
            enhanced_prompt = enhance_with_local_llm(
                kubin, hf_model_name, params["prompt"], llm_prompt
            )
        else:
            print(f"Using {ollama_model_name} via Ollama API")
            enhanced_prompt = enhance_with_llama_api(
                kubin, ollama_model_name, ollama_url, params["prompt"], llm_prompt
            )
        params["prompt"] = enhanced_prompt
        print(f"Enhanced prompt for {target} is: '{enhanced_prompt}'")


def setup(kubin):
    yaml_config = kubin.yaml_utils.YamlConfig(Path(__file__).parent.absolute())
    config = yaml_config.read()

    def get_local_llm_models():
        return config["hf-repo-models"].split(";")

    def get_ollama_models():
        return config["ollama-api-models"].split(";")

    def get_ollama_url():
        return config["ollama-url"]

    def get_tasks_list():
        return list(config["tasks"].keys())

    def get_prompt_for_task(model, task):
        prompt = config["tasks"][task].get(model, "None")

        if prompt == "None":
            prompt = config["tasks"][task]["default"]
        return prompt

    def llm_enhance_ui(target):
        target = gr.State(value=target)

        local_llm_models = get_local_llm_models()
        ollama_models = get_ollama_models()
        ollama_url = get_ollama_url()
        tasks_list = get_tasks_list()

        with gr.Column() as llm_selector_block:
            enable_llm_enhancer = gr.Checkbox(
                False, label="Enable", elem_classes=["llm-enhance-enable"]
            )

            with gr.Row():
                model_type = gr.Radio(
                    choices=[LLM_Type.HF_LOCAL.name, LLM_Type.OLLAMA_API.name],
                    value=LLM_Type.HF_LOCAL.name,
                    label="LLM type",
                    interactive=True,
                )

                hf_model_id = gr.Dropdown(
                    choices=[model for model in local_llm_models],
                    value=local_llm_models[0],
                    label="HF model ID",
                    allow_custom_value=True,
                    interactive=True,
                )

                ollama_model_id = gr.Dropdown(
                    choices=[model for model in ollama_models],
                    value=ollama_models[0],
                    label="Ollama model ID",
                    interactive=True,
                    allow_custom_value=True,
                    visible=False,
                )

                ollama_url = gr.Textbox(
                    value=ollama_url,
                    label="Ollama API URL",
                    visible=False,
                    max_lines=1,
                )

            with gr.Row():
                task = gr.Dropdown(
                    choices=tasks_list,
                    value=tasks_list[0],
                    label="Task",
                    interactive=True,
                )

            with gr.Row():
                prompt = gr.TextArea(
                    value=get_prompt_for_task(local_llm_models[0], tasks_list[0]),
                    label="Prompt",
                    interactive=True,
                    max_lines=5,
                    lines=5,
                )

            def on_type_change(model_type, hf_model_id, ollama_model_id, task):
                return [
                    gr.update(visible=model_type == LLM_Type.HF_LOCAL.name),
                    gr.update(visible=model_type == LLM_Type.OLLAMA_API.name),
                    gr.update(visible=model_type == LLM_Type.OLLAMA_API.name),
                    get_prompt_for_task(
                        (
                            hf_model_id
                            if model_type == LLM_Type.HF_LOCAL.name
                            else ollama_model_id
                        ),
                        task,
                    ),
                ]

            hf_model_id.change(
                fn=on_type_change,
                inputs=[model_type, hf_model_id, ollama_model_id, task],
                outputs=[hf_model_id, ollama_model_id, ollama_url, prompt],
                show_progress=False,
            )

            ollama_model_id.change(
                fn=on_type_change,
                inputs=[model_type, hf_model_id, ollama_model_id, task],
                outputs=[hf_model_id, ollama_model_id, ollama_url, prompt],
                show_progress=False,
            )

            task.change(
                fn=on_type_change,
                inputs=[model_type, hf_model_id, ollama_model_id, task],
                outputs=[hf_model_id, ollama_model_id, ollama_url, prompt],
                show_progress=False,
            )

            model_type.change(
                fn=on_type_change,
                inputs=[model_type, hf_model_id, ollama_model_id, task],
                outputs=[hf_model_id, ollama_model_id, ollama_url, prompt],
                show_progress=False,
            )

        llm_selector_block.elem_classes = ["kd-llm-enhancer-ui", "k-form"]
        return (
            llm_selector_block,
            enable_llm_enhancer,
            model_type,
            hf_model_id,
            ollama_model_id,
            ollama_url,
            prompt,
        )

    def settings_ui():
        def save_changes(hf_list, ollama_list, ollama_url):
            config["hf-repo-models"] = hf_list
            config["ollama-api-models"] = ollama_list
            config["ollama-url"] = ollama_url
            yaml_config.write(config)

        with gr.Column() as settings_block:
            llm_list = gr.Textbox(
                lambda: config["hf-repo-models"],
                label="HF Models",
                scale=0,
            )

            ollama_list = gr.Textbox(
                lambda: config["ollama-api-models"],
                label="Ollama Models",
                scale=0,
            )

            ollama_url = gr.Textbox(
                lambda: config["ollama-url"],
                label="Ollama API Url",
                scale=0,
            )

            save_btn = gr.Button("Save settings", size="sm", scale=0)
            save_btn.click(
                save_changes,
                inputs=[llm_list, ollama_list, ollama_url],
                outputs=[],
                queue=False,
            ).then(fn=None, _js=("(x) => kubin.notify.success('Settings saved')"))

        settings_block.elem_classes = ["k-form"]
        return settings_block

    return {
        "targets": targets,
        "title": title,
        "inject_ui": lambda target, kubin=kubin: llm_enhance_ui(target),
        "settings_ui": settings_ui,
        "inject_fn": lambda target, params, augmentations: enhance(
            kubin, target, params, *augmentations
        ),
        "inject_position": "before_params",
    }
