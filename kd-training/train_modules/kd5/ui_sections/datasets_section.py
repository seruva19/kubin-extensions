from dataclasses import dataclass

import gradio as gr

from ..ui_support import DATASET_FIELD_LABELS, dataset_table_headers


@dataclass
class DatasetSection:
    container: gr.Accordion
    table: gr.Dataframe
    add_dataset_btn: gr.Button
    remove_dataset_btn: gr.Button


def build_dataset_section(default_table) -> DatasetSection:
    initial_columns = len(default_table[0]) if default_table and default_table[0] else 2
    # Include headers as the first row for Gradio 3.50.2
    headers = dataset_table_headers(default_table)
    table_value = [headers] + default_table if default_table else [headers]

    with gr.Accordion("Datasets", open=True) as container:
        table = gr.Dataframe(
            value=table_value,
            datatype="str",
            row_count=(len(DATASET_FIELD_LABELS) + 1, "fixed"),  # +1 for header row
            col_count=(initial_columns, "dynamic"),
            wrap=True,
        )
        with gr.Row():
            add_dataset_btn = gr.Button("Add Dataset", variant="secondary", size="sm")
            remove_dataset_btn = gr.Button("Remove Last Dataset", variant="secondary", size="sm")

    container.elem_classes = ["kubin-accordion"]
    return DatasetSection(
        container=container,
        table=table,
        add_dataset_btn=add_dataset_btn,
        remove_dataset_btn=remove_dataset_btn
    )
