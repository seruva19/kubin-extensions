# Extensions for Kubin

## Description

Repository for storing "official" extensions for [Kubin](https://github.com/seruva19/kubin).

Initially, it contained only extensions that were somehow related to enhancing the capabilities of main Kubin functions (generating images with Kandinsky models), but it currently has a lot of stuff not directly related to Kandinsky (but necessarily connected with image generation). Some are outdated and not quite finished at all, and very few are actively maintained. Extensions are:

| <div style="width:200px">Name</div> | Description |
| --------------------- | - |
| <a id="kd-animation"></a>kd-animation | GUI wrapper for [Deforum-Kandinsky](https://github.com/ai-forever/deforum-kandinsky) |
| <a id="kd-bg-remover"></a>kd-bg-remover | GUI wrapper for [RMBG](https://huggingface.co/briaai/RMBG-1.4) and [BiRefNet](https://huggingface.co/ZhengPeng7/BiRefNet) background removal models |
| <a id="kd-flux"></a>kd-flux | Basic [🤗 diffusers](https://github.com/huggingface/diffusers)-based implementation of [Flux.1-Dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) T2I pipeline |
| <a id="kd-image-browser"></a>kd-image-browser | Tools for navigating output image folders |
| <a id="kd-image-editor"></a>kd-image-editor | GUI wrapper for [Filerobot Image Editor](https://github.com/scaleflex/filerobot-image-editor) |
| <a id="kd-image-tools"></a>kd-image-tools | Currently only allows image similarity search |
| <a id="kd-image-to-video"></a>kd-image-to-video | Based on [VideoCrafter](https://huggingface.co/VideoCrafter/Image2Video-512) model |
| <a id="kd-interrogator"></a>kd-interrogator | Contains GUI for single image/folder-targeted usage of image interrogation models: [CLIP Interrogator](https://github.com/pharmapsychotic/clip-interrogator) and VLM-based captioners: [CogVLM2](https://huggingface.co/THUDM/cogvlm2-llama3-chat-19B), [InternLM-XComposer2-4KHD](https://huggingface.co/internlm/internlm-xcomposer2-4khd-7b), [JoyCaption Pre-Alpha](https://huggingface.co/spaces/fancyfeast/joy-caption-pre-alpha), [JoyCaption Alpha One](https://huggingface.co/spaces/fancyfeast/joy-caption-alpha-one), [JoyCaption Alpha Two](https://huggingface.co/spaces/fancyfeast/joy-caption-alpha-two), [MiniCPM-V 2.6](https://huggingface.co/openbmb/MiniCPM-V-2_6), [Molmo-7B-O](https://huggingface.co/allenai/Molmo-7B-O-0924), [PaliGemma 2](https://huggingface.co/google/paligemma2-3b-ft-docci-448), [Pixtral-12B-Captioner-Relaxed](https://huggingface.co/Ertugrul/Pixtral-12B-Captioner-Relaxed), [Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct), [Qwen2-VL-7B-Captioner-Relaxed](https://huggingface.co/Ertugrul/Qwen2-VL-7B-Captioner-Relaxed) |
| <a id="kd-kwai-kolors"></a>kd-kwai-kolors | Basic implementation of [Kolors](https://github.com/Kwai-Kolors/Kolors) T2I pipeline |
| <a id="kd-llm-enhancer"></a>kd-llm-enhancer | Basic LLM-based prompt enhancer, based on [🤗 transformers](https://github.com/huggingface/transformers) and [Ollama API](https://github.com/ollama/ollama/blob/main/docs/api.md) |
| <a id="kd-mesh-gen"></a>kd-mesh-gen | Basic implementation of [Shap-E](https://github.com/openai/shap-e) I23D pipeline |
| <a id="kd-multi-view"></a>kd-multi-view | Basic implementation of [Zero123++](https://github.com/SUDO-AI-3D/zero123plus) "Image to Multi-view" pipeline |
| <a id="kd-networks"></a>kd-networks | Allows the use of Kandinsky 2.2 LoRA for inference |
| <a id="kd-pipeline-enhancer"></a>kd-pipeline-enhancer | Currently useless 🤔 |
| <a id="kd-pixart"></a>kd-pixart | Basic [🤗 diffusers](https://github.com/huggingface/diffusers)-based implementation of [PixArt-Sigma](https://github.com/PixArt-alpha/PixArt-sigma) T2I pipeline |
| <a id="kd-prompt-styles"></a>kd-prompt-styles | Enables auto-enhancing prompts with community-collected styles. |
| <a id="kd-sana"></a>kd-sana | Basic implementation of [SANA](https://github.com/NVlabs/Sana) T2I pipeline |
| <a id="kd-segmentation"></a>kd-segmentation | GUI wrapper for [Segment Anything](https://github.com/facebookresearch/segment-anything). Was intended for auto-extraction of inpainting masks and custom ADetailer implementation for Kandinsky, but was not finished 😔 |
| <a id="kd-stable-cascade"></a>kd-stable-cascade | Basic [🤗 diffusers](https://github.com/huggingface/diffusers)-based implementation of [Stable Cascade](https://github.com/Stability-AI/StableCascade) T2I pipeline |
| <a id="kd-switti"></a>kd-switti | Basic implementation of the T2I pipeline for [Switti](https://github.com/yandex-research/switti) |
| <a id="kd-training"></a>kd-training | GUI wrapper for some Kandinsky training scripts (K2.1 fine-tuning/K2.2 LoRA) |
| <a id="kd-upscaler"></a>kd-upscaler | Tools for upscaling, currently only [Real-ESRGAN](https://github.com/ai-forever/Real-ESRGAN) and [KandiSuperRes](https://github.com/ai-forever/KandiSuperRes) are supported |
| <a id="kd-video"></a>kd-video | GUI for a consumer-friendly (24Gb VRAM) implementation of [Kandinsky Video](https://github.com/ai-forever/KandinskyVideo) T2V/I2V pipelines. The low-VRAM pipeline for KV1.1 is still flawed and outputs noise 🙄 |

## Installation

Clone this repo into the "extensions" folder in Kubin root. More info [here](https://github.com/seruva19/kubin/wiki/Docs).

