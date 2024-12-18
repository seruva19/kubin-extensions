import gradio as gr
import os
import torch


def networks_converter_ui(kubin):
    with gr.Row() as nn_converters_block:
        with gr.Accordion("Weights conversion", open=True) as network_conversion:
            with gr.Row():
                conversion_from = gr.Dropdown(
                    choices=["bin"],
                    type="value",
                    value="bin",
                    label="Source format",
                )
                conversion_to = gr.Dropdown(
                    choices=["safetensors"],
                    type="value",
                    value="safetensors",
                    label="Target format",
                )

            with gr.Column():
                with gr.Row():
                    source_path = gr.Text(
                        "", label="Source path", info="Source network"
                    )
                    target_path = gr.Text(
                        "",
                        label="Target path",
                        info="Leaving empty will save to the same folder and the same name",
                    )

                with gr.Row():
                    convert_nn_btn = gr.Button("🔀 Convert", scale=0)

            def convert_network(convert_from, convert_to, source_path, target_path):
                if convert_from == "bin" and convert_to == "safetensors":
                    if target_path == "":
                        target_folder = os.path.dirname(source_path)
                        filename = os.path.basename(source_path)
                        filename_without_extension = os.path.splitext(filename)[0]
                        target_path = os.path.join(
                            target_folder, filename_without_extension + ".safetensors"
                        )

                    kubin.nn_utils.convert_pt_to_sft(source_path, target_path)

            kubin.ui_utils.click_and_disable(
                convert_nn_btn,
                convert_network,
                [conversion_from, conversion_to, source_path, target_path],
                None,
            ).then(
                fn=None,
                _js="_ => kubin.notify.success('Conversion completed')",
                inputs=None,
                outputs=None,
            )

        with gr.Accordion("Weights type", open=True) as dtype_conversion:
            with gr.Row():
                torch_type = gr.Dropdown(
                    choices=[
                        "torch.float32",
                        "torch.float",
                        "torch.float64",
                        "torch.double",
                        "torch.float16",
                        "torch.half",
                        "torch.bfloat16",
                        "torch.complex32",
                        "torch.chalf",
                        "torch.complex64",
                        "torch.cfloat",
                        "torch.complex128",
                        "torch.cdouble",
                        "torch.uint8",
                        "torch.uint16",
                        "torch.uint32",
                        "torch.uint64",
                        "torch.int8",
                        "torch.int16",
                        "torch.short",
                        "torch.int32",
                        "torch.int",
                        "torch.int64",
                        "torch.long",
                        "torch.bool",
                        "torch.quint8",
                        "torch.qint8",
                        "torch.qint32",
                        "torch.quint4x2",
                        "torch.float8_e4m3fn",
                        "torch.float8_e5m2",
                    ],
                    type="value",
                    info="PyTorch tensor type",
                    value="torch.bfloat16",
                    label="Type",
                )
            with gr.Row():
                source_tensors_path = gr.Text(
                    "", label="Source weights path", info="Source weights"
                )
                target_tensors_path = gr.Text(
                    "",
                    label="Target path",
                    info="Leaving empty will save to the same folder",
                )

            with gr.Row():
                convert_dtype_btn = gr.Button("🔀 Convert", scale=0)

            def convert_tensor_type(convert_to_dtype, source_path, target_path):
                if target_path == "":
                    target_folder = os.path.dirname(source_path)
                    filename = os.path.basename(source_path)
                    filename_without_extension = os.path.splitext(filename)[0]
                    extension = os.path.splitext(filename)[1]
                    target_path = os.path.join(
                        target_folder,
                        filename_without_extension + f"-{convert_to_dtype}{extension}",
                    )

                    dtype = getattr(torch, convert_to_dtype.split(".")[-1])
                    kubin.nn_utils.convert_torch_dtype(source_path, target_path, dtype)

            kubin.ui_utils.click_and_disable(
                convert_dtype_btn,
                convert_tensor_type,
                [torch_type, source_tensors_path, target_tensors_path],
                None,
            ).then(
                fn=None,
                _js="_ => kubin.notify.success('Conversion completed')",
                inputs=None,
                outputs=None,
            )

    network_conversion.elem_classes = dtype_conversion.elem_classes = [
        "kubin-accordion"
    ]
    return nn_converters_block
