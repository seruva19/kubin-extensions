# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# This file is modified from https://github.com/haotian-liu/LLaVA/

# seruva19: this source code was adopted from https://github.com/efogdev/apollo/

CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."


# Model Constants
IGNORE_INDEX = -100
X_TOKEN_INDEX = -200
X_TOKEN = {"image": "<|image_token|>", "video": "<|video_token|>"}
X_PATCH_TOKEN = {"image": "<|image_patch|>", "video": "<|video_patch|>"}
X_START_TOKEN = {"image": "<|image_start|>", "video": "<|video_start|>"}
X_END_TOKEN = {"image": "<|image_end|>", "video": "<|video_end|>"}
