import torch
import os
from diffusers.models.unet_2d_condition import UNet2DConditionModel


def set_params(kubin, params, decoder, model, enhancer_info):
    task = params.get(".ui-task", "none")
    model_config = model.config
    unet = decoder.unet

    tune_freeu(kubin, model_config, unet, params, task, enhancer_info["freeu"])


def tune_freeu(kubin, model_config, unet, params, task, freeu):
    params_session = params[".session"]
    current_freeu = freeu.get(f"{task}-{params_session}", None)

    unet_has_freeu = model_config.get(".freeu", None) is not None
    freeu_enabled = current_freeu is not None and current_freeu["enabled"] == True

    if unet_has_freeu and not freeu_enabled:
        kubin.elog(f"disabled free_u")
        unet.disable_freeu()
        model_config.pop(".freeu")

    if freeu_enabled:
        s1 = current_freeu["s1"]
        s2 = current_freeu["s2"]
        b1 = current_freeu["b1"]
        b2 = current_freeu["b2"]

        kubin.elog(f"enabled free_u with params: s1={s1}, s2={s2}, b1={b1}, b2={b2}")
        unet.enable_freeu(s1, s2, b1, b2)

        model_config[".freeu"] = current_freeu
