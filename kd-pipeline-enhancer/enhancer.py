import gradio as gr


def enhancer_ui(kubin, target, enhancer_info):
    with gr.Column() as enhancer_ui_block:
        with gr.Accordion("Free-U") as freeu_block:
            with gr.Column() as freeu_panel:
                freeu_panel.elem_classes = ["k-form"]

                enable_freeu_check = gr.Checkbox(
                    False,
                    label="Enable FreeU",
                    elem_classes=["pipeline-enable-freeu"],
                )

                with gr.Row():
                    freeu_s1 = gr.Number(
                        value=0.9, minimum=0, maximum=1, step=0.1, label="S1"
                    )
                    freeu_s2 = gr.Number(
                        value=0.2, minimum=0, maximum=1, step=0.1, label="S2"
                    )
                    freeu_b1 = gr.Number(
                        value=1.3, minimum=1, maximum=2, step=0.1, label="B1"
                    )
                    freeu_b2 = gr.Number(
                        value=1.4, minimum=1, maximum=2, step=0.1, label="B2"
                    )

            def enable_freeu(session, enable_freeu, s1, s2, b1, b2):
                freeu_state = {
                    "enabled": enable_freeu,
                    "s1": s1,
                    "s2": s2,
                    "b1": b1,
                    "b2": b2,
                }

                task = kubin.env_utils.map_target_to_task(target)
                enhancer_info["freeu"][f"{task}-{session}"] = freeu_state

            session = gr.Textbox(visible=False)
            enable_freeu_check.change(
                _js="(...args) => [window._kubinSession, ...args.slice(1)]",
                fn=enable_freeu,
                inputs=[
                    session,
                    enable_freeu_check,
                    freeu_s1,
                    freeu_s2,
                    freeu_b1,
                    freeu_b2,
                ],
                outputs=[],
            )
            freeu_s1.change(
                _js="(...args) => [window._kubinSession, ...args.slice(1)]",
                fn=enable_freeu,
                inputs=[
                    session,
                    enable_freeu_check,
                    freeu_s1,
                    freeu_s2,
                    freeu_b1,
                    freeu_b2,
                ],
                outputs=[],
            )
            freeu_s2.change(
                _js="(...args) => [window._kubinSession, ...args.slice(1)]",
                fn=enable_freeu,
                inputs=[
                    session,
                    enable_freeu_check,
                    freeu_s1,
                    freeu_s2,
                    freeu_b1,
                    freeu_b2,
                ],
                outputs=[],
            )
            freeu_b1.change(
                _js="(...args) => [window._kubinSession, ...args.slice(1)]",
                fn=enable_freeu,
                inputs=[
                    session,
                    enable_freeu_check,
                    freeu_s1,
                    freeu_s2,
                    freeu_b1,
                    freeu_b2,
                ],
                outputs=[],
            )
            freeu_b2.change(
                _js="(...args) => [window._kubinSession, ...args.slice(1)]",
                fn=enable_freeu,
                inputs=[
                    session,
                    enable_freeu_check,
                    freeu_s1,
                    freeu_s2,
                    freeu_b1,
                    freeu_b2,
                ],
                outputs=[],
            )

        with gr.Accordion("Samplers") as samplers_block:
            pass

    return enhancer_ui_block
