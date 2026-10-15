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

"""CPU checks for Cosmos3-H-Dreams offline inputs."""

import numpy as np
import pytest
from cosmos3_h_dreams.inputs import ACTION_DIM, load_actions

pytestmark = pytest.mark.ci_cpu


def test_load_actions_pads_normalized_actions(tmp_path) -> None:
    source = np.arange(8 * ACTION_DIM, dtype=np.float32).reshape(8, ACTION_DIM)
    path = tmp_path / "actions.npy"
    np.save(path, source)

    actual = load_actions(path).numpy()

    assert actual.shape == (8, 64)
    np.testing.assert_array_equal(actual[:, :ACTION_DIM], source)
    assert not np.count_nonzero(actual[:, ACTION_DIM:])


def test_load_actions_applies_checkpoint_normalizer_before_padding(tmp_path) -> None:
    source = np.full((4, ACTION_DIM), 5.0, dtype=np.float32)
    actions = tmp_path / "actions.npy"
    normalizer = tmp_path / "normalizer.npz"
    np.save(actions, source)
    np.savez(
        normalizer,
        offset=np.ones(ACTION_DIM, dtype=np.float32),
        scale=np.full(ACTION_DIM, 2.0, dtype=np.float32),
    )

    actual = load_actions(actions, normalizer).numpy()

    np.testing.assert_array_equal(actual[:, :ACTION_DIM], np.full((4, ACTION_DIM), 2.0))
    assert not np.count_nonzero(actual[:, ACTION_DIM:])


@pytest.mark.parametrize(
    "shape", [(4, ACTION_DIM - 1), (4, ACTION_DIM + 1), (1, 4, ACTION_DIM)]
)
def test_load_actions_rejects_wrong_shape(tmp_path, shape) -> None:
    path = tmp_path / "actions.npy"
    np.save(path, np.zeros(shape, dtype=np.float32))

    with pytest.raises(ValueError, match=rf"shape \[N, {ACTION_DIM}\]"):
        load_actions(path)


def test_load_actions_rejects_partial_latent_frame(tmp_path) -> None:
    path = tmp_path / "actions.npy"
    np.save(path, np.zeros((6, ACTION_DIM), dtype=np.float32))

    with pytest.raises(ValueError, match="divisible by 4"):
        load_actions(path)
