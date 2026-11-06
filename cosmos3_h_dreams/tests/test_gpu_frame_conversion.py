# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Same-pixel CUDA/CPU conversion checks; no model weights or downloads."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from cosmos3_h_dreams.headset_server import MjpegFrames

pytestmark = pytest.mark.ci_gpu


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["tchw", "thwc"])
def test_gpu_frame_conversion_matches_host(dtype, layout):
    assert torch.cuda.is_available(), "CUDA required; no CPU fallback for this test"
    # Includes clipping, zero and integer quantization boundaries at display scale.
    values = torch.cat((torch.linspace(-1.1, 1.1, 8192), torch.arange(256) / 127.5 - 1))
    video = values.repeat(3).reshape(1, 3, 64, 132).to(device="cuda", dtype=dtype)
    if layout == "thwc":
        video = video.permute(0, 2, 3, 1).contiguous()
    result = SimpleNamespace(video_chunk=video, layout=layout)
    baseline = MjpegFrames().prepare_result_frames(result)
    candidate = MjpegFrames(gpu_frame_conversion=True).prepare_result_frames(result)
    np.testing.assert_array_equal(candidate, baseline)
