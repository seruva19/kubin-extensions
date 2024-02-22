from enhancer import enhancer_ui
from pipeline_proc import set_params
import gradio as gr
from pathlib import Path

dir = Path(__file__).parent.absolute()
title = "Enhancer"

enhancer_info = {"freeu": {}}


def setup(kubin):
    def on_hook(hook, **kwargs):
        if hook == kubin.params.HOOK.BEFORE_PREPARE_DECODER:
            model = kwargs["model"]
            if hasattr(model, "config"):
                set_params(
                    kubin,
                    kwargs["params"],
                    kwargs["decoder"],
                    model,
                    enhancer_info,
                )
            else:
                kubin.log(
                    f"Selected pipeline is not supported by '{title}' extension extension, 'BEFORE_PREPARE_PARAMS' hook is ignored."
                )

    return {
        "targets": ["t2i", "i2i", "mix", "inpaint", "outpaint"],
        "title": title,
        "inject_ui": lambda target, kubin=kubin: enhancer_ui(
            kubin, target, enhancer_info
        ),
        "hook_fn": on_hook,
        "supports": ["diffusers-kd22"],
        "inject_position": "before_params",
    }
