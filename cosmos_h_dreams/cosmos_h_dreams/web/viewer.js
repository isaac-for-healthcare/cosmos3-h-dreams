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
let peer = null;
video.onplaying = () => { status.textContent = "Live Quest video"; };

async function connect() {
  if (video.dataset.transport === "mjpeg") {
    video.src = `/video?t=${Date.now()}`;
    status.textContent = "Live view · Waiting for Quest frames…";
    video.onerror = () => {
      status.textContent = "Video connection lost · Reconnecting…";
      setTimeout(() => { void connect().catch(showError); }, 1000);
    };
    return;
  }
  const connection = new RTCPeerConnection();
  peer = connection;
  connection.addTransceiver("video", { direction: "recvonly" });
  connection.ontrack = (event) => {
    video.srcObject = event.streams[0] || new MediaStream([event.track]);
    void video.play().catch((error) => { status.textContent = `Video playback: ${error.message}`; });
  };
  connection.onconnectionstatechange = () => {
    if (peer !== connection) return;
    status.textContent = connection.connectionState === "connected"
      ? "Live view · Waiting for Quest frames…"
      : `Video: ${connection.connectionState}`;
    if (connection.connectionState === "failed" || connection.connectionState === "disconnected") {
      connection.close();
      setTimeout(() => { if (peer === connection) void connect().catch(showError); }, 1000);
    }
  };
  await connection.setLocalDescription(await connection.createOffer());
  if (connection.iceGatheringState !== "complete") {
    await new Promise((resolve) => {
      const done = () => {
        if (connection.iceGatheringState === "complete") {
          connection.removeEventListener("icegatheringstatechange", done);
          resolve();
        }
      };
      connection.addEventListener("icegatheringstatechange", done);
      done();
    });
  }
  const response = await fetch("/viewer/offer", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(connection.localDescription),
  });
  if (!response.ok) throw new Error(`WebRTC offer failed: ${response.status} ${await response.text()}`);
  await connection.setRemoteDescription(await response.json());
}

function showError(error) {
  if (peer) peer.close();
  status.textContent = error.message;
  setTimeout(() => { void connect().catch(showError); }, 1000);
}

void connect().catch(showError);
void refreshModelStatus();
