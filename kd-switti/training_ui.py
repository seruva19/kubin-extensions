import gc
import shutil
import subprocess
import sys
import tempfile
from uuid import uuid4
import gradio as gr
import os
import re
from pathlib import Path
import os, torch


def training_block(kubin):
    with gr.Row() as switti_training_block:
        pass

    return switti_training_block
