// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Reads headset controllers in an immersive WebXR session and streams them to the
// demo client as `xr_frame` messages.
//
// The schema is the one documented in headset_controller.py and implemented by
// synthetic_source.py. Keeping to it is the whole contract: anything that
// produces these messages can drive the demo.
//
// Immersive VR rather than AR: the headset usually sits on a desk rather than on
// a head, so passthrough buys nothing and AR is the slower session to negotiate.

"use strict";

(() => {
  const WS_PATH = "/ws";
  const RECONNECT_MIN_MS = 500;
  const RECONNECT_MAX_MS = 5000;

  // Controllers report a grip pose ahead of the hand. Shifting the readout back
  // along the controller's local Z puts it near the wrist pivot, so rotating the
  // wrist produces almost no translation. Without this a twist reads as a lunge.
  const WRIST_OFFSET_M = [0, 0, 0.05];

  const el = (id) => document.getElementById(id);
  const enterButton = el("enter-vr");
  const xrState = el("xr-state");
  const wsState = el("ws-state");
  const sentCount = el("sent");
  const logView = el("log");

  let socket = null;
  let reconnectMs = RECONNECT_MIN_MS;
  let session = null;
  let referenceSpace = null;
  let sent = 0;

  function log(message) {
    const stamp = new Date().toLocaleTimeString();
    logView.textContent = `${stamp}  ${message}\n${logView.textContent}`.slice(0, 4000);
  }

  // ---- transport ---------------------------------------------------------

  function connect() {
    const scheme = location.protocol === "https:" ? "wss:" : "ws:";
    socket = new WebSocket(`${scheme}//${location.host}${WS_PATH}`);

    socket.onopen = () => {
      reconnectMs = RECONNECT_MIN_MS;
      wsState.textContent = "connected";
      log("client connected");
      // Tells the demo the page is alive, so a stalled startup can name the
      // step it is stuck on instead of just timing out.
      report(session ? "session" : "loaded");
    };
    socket.onclose = () => {
      wsState.textContent = "reconnecting";
      // The demo client restarting must not require walking over to the headset
      // and reloading the page, which is what a connect-once page costs.
      setTimeout(connect, reconnectMs);
      reconnectMs = Math.min(reconnectMs * 2, RECONNECT_MAX_MS);
    };
    socket.onerror = () => {
      wsState.textContent = "error";
    };
  }

  function send(payload) {
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify(payload));
      return true;
    }
    return false;
  }

  function report(stage, detail) {
    send({ type: "status", stage, detail: detail || "" });
  }

  // ---- pose maths --------------------------------------------------------

  // Rotate a vector by an xyzw quaternion: v + 2 * (q.xyz x (q.xyz x v + w*v)).
  function rotate(vector, q) {
    const [x, y, z, w] = q;
    const [vx, vy, vz] = vector;
    const tx = 2 * (y * vz - z * vy);
    const ty = 2 * (z * vx - x * vz);
    const tz = 2 * (x * vy - y * vx);
    return [
      vx + w * tx + (y * tz - z * ty),
      vy + w * ty + (z * tx - x * tz),
      vz + w * tz + (x * ty - y * tx),
    ];
  }

  function readController(source, frame) {
    const space = source.gripSpace || source.targetRaySpace;
    if (!space) return null;
    const pose = frame.getPose(space, referenceSpace);
    if (!pose) return null;

    const o = pose.transform.orientation;
    const p = pose.transform.position;
    const orientation = [o.x, o.y, o.z, o.w];
    const shift = rotate(WRIST_OFFSET_M, orientation);
    const pad = source.gamepad;
    return {
      position: [p.x + shift[0], p.y + shift[1], p.z + shift[2]],
      orientation,
      // Serialized whole so the demo can apply the xr-standard button indices
      // itself rather than this page deciding what each button means.
      buttons: pad ? pad.buttons.map((b) => ({ p: b.pressed, t: b.touched, v: b.value })) : [],
      axes: pad ? Array.from(pad.axes) : [],
    };
  }

  function onFrame(_time, frame) {
    if (!session) return;
    session.requestAnimationFrame(onFrame);

    const controllers = {};
    for (const source of session.inputSources) {
      if (source.handedness !== "left" && source.handedness !== "right") continue;
      const reading = readController(source, frame);
      // An absent controller is simply omitted; the demo holds its last pose
      // and marks it untracked, which reads better than a teleport on return.
      if (reading) controllers[source.handedness] = reading;
    }
    if (Object.keys(controllers).length === 0) return;

    if (send({ type: "xr_frame", t_client: performance.now(), controllers })) {
      sent += 1;
      if (sent % 30 === 0) sentCount.textContent = String(sent);
    }
  }

  // ---- session -----------------------------------------------------------

  async function enter() {
    if (session) {
      await session.end();
      return;
    }
    let setupStage = "WebGL2";
    try {
      const canvas = document.createElement("canvas");
      const gl = canvas.getContext("webgl2", { xrCompatible: true });
      if (!gl) throw new Error("WebGL2 unavailable");
      setupStage = "requestSession(immersive-vr)";
      session = await navigator.xr.requestSession("immersive-vr");
      setupStage = "XRWebGLLayer";
      session.updateRenderState({ baseLayer: new XRWebGLLayer(session, gl) });
      setupStage = "requestReferenceSpace(local-floor)";
      try {
        referenceSpace = await session.requestReferenceSpace("local-floor");
      } catch (_error) {
        setupStage = "requestReferenceSpace(local)";
        referenceSpace = await session.requestReferenceSpace("local");
      }
      session.addEventListener("end", () => {
        session = null;
        referenceSpace = null;
        xrState.textContent = "ready";
        enterButton.textContent = "Start Teleop";
        log("session ended");
        report("ended");
      });
      xrState.textContent = "streaming";
      enterButton.textContent = "Stop Teleop";
      log("session started");
      report("session");
      session.requestAnimationFrame(onFrame);
    } catch (error) {
      session = null;
      xrState.textContent = "failed";
      const detail = `${setupStage}: ${error.name || "Error"}: ${error.message || String(error)}`;
      log(`could not start: ${detail}`);
      report("failed", detail);
    }
  }

  async function init() {
    connect();
    if (!navigator.xr) {
      xrState.textContent = "unsupported";
      enterButton.textContent = "WebXR unavailable";
      return;
    }
    const supported = await navigator.xr.isSessionSupported("immersive-vr").catch(() => false);
    if (!supported) {
      xrState.textContent = "unsupported";
      enterButton.textContent = "Immersive VR unavailable";
      return;
    }
    xrState.textContent = "ready";
    enterButton.disabled = false;
    enterButton.addEventListener("click", enter);
  }

  init();
})();
