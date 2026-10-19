// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

"use strict";

const video = document.getElementById("video");
const status = document.getElementById("status");
const modelStatus = document.getElementById("model-status");
async function refreshModelStatus() {
  try {
    const response = await fetch("/healthz", { cache: "no-store" });
    if (!response.ok) throw new Error("Model status unavailable");
    const health = await response.json();
    modelStatus.textContent = health.model_status;
    updateScenes(health);
  } catch (_) {
    modelStatus.textContent = "Model status unavailable · Reconnecting…";
  } finally {
    setTimeout(refreshModelStatus, 1000);
  }
}
video.onplaying = () => { status.textContent = "Live headset video"; };

function connect() {
  video.src = `/video?t=${Date.now()}`;
  status.textContent = "Live view · Waiting for headset frames…";
  video.onerror = () => {
    status.textContent = "Video connection lost · Reconnecting…";
    setTimeout(connect, 1000);
  };
}

connect();
void refreshModelStatus();
