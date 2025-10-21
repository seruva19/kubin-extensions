# This source code is licensed under the S-Lab License 1.0 found in the
# LICENSE file in the current directory's parent directory.
"""
The code has been adopted from FAST-VQA-and-FasterVQA
(https://github.com/VQAssessment/FAST-VQA-and-FasterVQA/blob/dev/fastvqa/models/__init__.py)
"""

from .swin_backbone import SwinTransformer3D as VQABackbone
from .swin_backbone import SwinTransformer2D as IQABackbone
from .head import VQAHead, IQAHead, VARHead
from .swin_backbone import swin_3d_tiny, swin_3d_small
from .conv_backbone import convnext_3d_tiny, convnext_3d_small
from .xclip_backbone import build_x_clip_model
from .evaluator import BaseEvaluator, BaseImageEvaluator, DiViDeAddEvaluator

__all__ = [
    "VQABackbone",
    "IQABackbone",
    "VQAHead",
    "IQAHead",
    "VARHead",
    "BaseEvaluator",
    "BaseImageEvaluator",
    "DiViDeAddEvaluator",
]
