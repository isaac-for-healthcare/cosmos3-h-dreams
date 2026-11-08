# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Bounded keyboard instrument poses using the Cosmos3 28D action contract."""

import threading
import time

import numpy as np
import torch

from cosmos3_h_dreams.inputs import normalize_and_pad_actions


class KeyboardActionSource:
    """Generate pose targets from held keys; stale input stops movement."""

    KEYS = frozenset({"w", "a", "s", "d", "q", "e", "z", "x", "arrowup", "arrowdown", "arrowleft", "arrowright", "pageup", "pagedown"})

    def __init__(self, *, fps, normalizer=None, jaw_open=1.0, jaw_closed=-2.0, **_kwargs):
        self.fps = fps
        self.normalizer = normalizer
        self.jaw_open, self.jaw_closed = jaw_open, jaw_closed
        self._lock = threading.Lock()
        self._position = np.zeros((2, 3), dtype=np.float32)
        self._keys = set()
        self._pending = set()
        self._updated = 0.0
        self._restart = False

    def start(self):
        """Start with neutral instrument targets."""
        self.reset_calibration()

    def set_keys(self, keys):
        """Accept only a bounded list of known browser keys."""
        if not isinstance(keys, list) or len(keys) > len(self.KEYS) or any(not isinstance(key, str) or key not in self.KEYS for key in keys):
            raise ValueError("Unknown keyboard control")
        with self._lock:
            self._pending.update(set(keys) - self._keys)
            self._keys = set(keys)
            self._updated = time.monotonic()

    def next_actions(self, rows):
        """Build 28D poses and pad to the model's 64D action input."""
        actions = np.zeros((rows, 28), dtype=np.float32)
        with self._lock:
            keys = self._keys if time.monotonic() - self._updated < 1.0 else set()
            # Preserve completed taps, but do not replay an expired held key.
            pending = self._pending if keys else self._pending - self._keys
            self._pending = set()
            for arm, (offset, bindings, jaw_key) in enumerate((
                (0, (("d", "a"), ("s", "w"), ("e", "q")), "z"),
                (14, (("arrowright", "arrowleft"), ("arrowdown", "arrowup"), ("pageup", "pagedown")), "x"),
            )):
                for row in range(rows):
                    active = keys | pending if row == 0 else keys
                    delta = np.array([int(plus in active) - int(minus in active) for plus, minus in bindings]) * (0.1 / self.fps)
                    self._position[arm] = np.clip(self._position[arm] + delta, -0.5, 0.5)
                    actions[row, offset:offset + 3] = self._position[arm]
                    actions[row, offset + 9] = self.jaw_closed if jaw_key in active else self.jaw_open
                actions[:, offset + 3:offset + 9] = (1, 0, 0, 0, 1, 0)
        padded = normalize_and_pad_actions(actions, self.normalizer)
        padded[:, [9, 10, 23, 24]] = torch.from_numpy(actions[:, [9, 10, 23, 24]].copy())
        return padded

    def calibration_state(self):
        return True, "Keyboard ready", 1.0

    def reset_calibration(self):
        with self._lock:
            self._position.fill(0)
            self._keys.clear()
            self._pending.clear()

    def request_restart(self):
        self._restart = True

    def take_restart_request(self):
        restart, self._restart = self._restart, False
        return restart

    def set_normalizer(self, normalizer):
        self.normalizer = normalizer

    def close(self):
        self.reset_calibration()
