// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

"use strict";

// Video and the controller WebSocket share the same HTTPS origin.
// The server converts these poses to this checkpoint's fixed-anchor 28D actions.
const video = document.getElementById("video");
const status = document.getElementById("status");
const modelStatus = document.getElementById("model-status");
const statusCanvas = document.createElement("canvas");
statusCanvas.width = 1024;
statusCanvas.height = 128;
let statusTextureText = null;
let statusShownAt = 0;
const enterButton = document.getElementById("enter-vr");
const resetButton = document.getElementById("reset");
let socket = null;
let xrSession = null;
let xrSpace = null;
let gl = null;
let quad = null;
let statusQuad = null;
let resetHeldSince = null;
let resetSent = false;

function show(message) { status.textContent = message; }
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
function send(payload) {
  if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify(payload));
}

function connectVideo() {
  video.src = `/video?t=${Date.now()}`;
  video.onerror = () => show("Video connection lost · Reconnect the page");
  show("Controller: connected · Video: MJPEG");
}

function connect() {
  const scheme = location.protocol === "https:" ? "wss:" : "ws:";
  socket = new WebSocket(`${scheme}//${location.host}/ws`);
  socket.onopen = () => {
    show("Controller connected · Connecting video…");
    connectVideo();
  };
  socket.onmessage = (event) => {
    try {
      const message = JSON.parse(event.data);
      if (message.type === "error") show(message.message);
    } catch (_) { /* Ignore malformed diagnostics. */ }
  };
  socket.onclose = () => {
    show("Connection lost · Reconnecting…");
    video.removeAttribute("src");
    setTimeout(connect, 1000);
  };
}

function matMul(a, b) {
  const out = new Float32Array(16);
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++)
    for (let k = 0; k < 4; k++) out[c * 4 + r] += a[k * 4 + r] * b[c * 4 + k];
  return out;
}

function shader(type, source) {
  const item = gl.createShader(type);
  gl.shaderSource(item, source);
  gl.compileShader(item);
  if (!gl.getShaderParameter(item, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(item));
  return item;
}

function makeQuad() {
  const program = gl.createProgram();
  gl.attachShader(program, shader(gl.VERTEX_SHADER, `
    attribute vec2 point; attribute vec2 uv; uniform mat4 transform;
    varying vec2 texcoord;
    void main() { texcoord = uv; gl_Position = transform * vec4(point, 0.0, 1.0); }
  `));
  gl.attachShader(program, shader(gl.FRAGMENT_SHADER, `
    precision mediump float; varying vec2 texcoord; uniform sampler2D picture;
    void main() { gl_FragColor = texture2D(picture, texcoord); }
  `));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
  const buffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
    -0.50, -0.32, 0, 1,  0.50, -0.32, 1, 1,
    -0.50,  0.32, 0, 0,  0.50,  0.32, 1, 0,
  ]), gl.STATIC_DRAW);
  const texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([0, 0, 0, 255]));
  return { program, buffer, texture,
    point: gl.getAttribLocation(program, "point"),
    uv: gl.getAttribLocation(program, "uv"),
    transform: gl.getUniformLocation(program, "transform"),
    picture: gl.getUniformLocation(program, "picture") };
}

function renderVideo(viewer, layer) {
  if (!viewer || !quad) return;
  const hasFrame = video.naturalWidth > 0;
  if (hasFrame) {
    gl.bindTexture(gl.TEXTURE_2D, quad.texture);
    try { gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, video); }
    catch (_) { /* Keep the previous frame during a decoder transition. */ }
  }
  const ahead = new Float32Array([1, 0, 0, 0,  0, 1, 0, 0,  0, 0, 1, 0,  0, 0, -1.2, 1]);
  const model = matMul(viewer.transform.matrix, ahead);
  if (statusTextureText !== modelStatus.textContent) {
    const context = statusCanvas.getContext("2d");
    context.fillStyle = "#10131a";
    context.fillRect(0, 0, statusCanvas.width, statusCanvas.height);
    context.fillStyle = "#e6e9ef";
    context.font = "40px system-ui, sans-serif";
    context.textAlign = "center";
    context.textBaseline = "middle";
    context.fillText(modelStatus.textContent, 512, 64, 980);
    gl.bindTexture(gl.TEXTURE_2D, statusQuad.texture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, statusCanvas);
    statusTextureText = modelStatus.textContent;
    statusShownAt = performance.now();
  }
  const panel = matMul(viewer.transform.matrix, new Float32Array([
    1, 0, 0, 0,  0, 0.2, 0, 0,  0, 0, 1, 0,  0, -0.42, -1.2, 1,
  ]));
  for (const [item, transform] of [[quad, model], [statusQuad, panel]]) {
    if (item === statusQuad && ["Model compiled", "Model ready"].includes(modelStatus.textContent)
        && performance.now() - statusShownAt >= 2000) continue;
    gl.useProgram(item.program);
    gl.bindBuffer(gl.ARRAY_BUFFER, item.buffer);
    gl.enableVertexAttribArray(item.point);
    gl.vertexAttribPointer(item.point, 2, gl.FLOAT, false, 16, 0);
    gl.enableVertexAttribArray(item.uv);
    gl.vertexAttribPointer(item.uv, 2, gl.FLOAT, false, 16, 8);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, item.texture);
    gl.uniform1i(item.picture, 0);
    for (const view of viewer.views) {
      const viewport = layer.getViewport(view);
      gl.viewport(viewport.x, viewport.y, viewport.width, viewport.height);
      gl.uniformMatrix4fv(item.transform, false,
        matMul(view.projectionMatrix, matMul(view.transform.inverse.matrix, transform)));
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    }
  }
}

function readController(source, frame) {
  const space = source.gripSpace;
  if (!space) return null;
  const pose = frame.getPose(space, xrSpace);
  if (!pose) return null;
  const p = pose.transform.position, o = pose.transform.orientation;
  const orientation = [o.x, o.y, o.z, o.w];
  const gamepad = source.gamepad;
  return {
    position: [p.x, p.y, p.z],
    orientation,
    grip: Number(gamepad?.buttons[1]?.pressed ?? false),
    buttons: gamepad ? gamepad.buttons.map((button) => ({ p: button.pressed, v: button.value })) : [],
    axes: gamepad ? Array.from(gamepad.axes) : [],
  };
}

function onFrame(t, frame) {
  if (!xrSession) return;
  xrSession.requestAnimationFrame(onFrame);
  const layer = xrSession.renderState.baseLayer;
  gl.bindFramebuffer(gl.FRAMEBUFFER, layer.framebuffer);
  gl.clearColor(0, 0, 0, 1);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  renderVideo(frame.getViewerPose(xrSpace), layer);

  const controllers = {};
  for (const source of xrSession.inputSources) {
    if (source.handedness !== "left" && source.handedness !== "right") continue;
    const reading = readController(source, frame);
    if (reading) controllers[source.handedness] = reading;
  }
  if (controllers.left || controllers.right)
    send({ type: "xr_frame", t_client: t, controllers });

  const b = controllers.right?.buttons?.[5]?.p;
  if (b) {
    if (resetHeldSince === null) resetHeldSince = t;
    if (!resetSent && t - resetHeldSince >= 1000) {
      send({ type: "reset" });
      resetSent = true;
    }
  } else {
    resetHeldSince = null;
    resetSent = false;
  }
}

async function enterVr() {
  if (xrSession) { await xrSession.end(); return; }
  const nextSession = await navigator.xr.requestSession("immersive-vr", { optionalFeatures: ["local-floor"] });
  try {
    const canvas = document.createElement("canvas");
    gl = canvas.getContext("webgl", { xrCompatible: true, alpha: false });
    if (!gl) throw new Error("WebGL unavailable");
    await gl.makeXRCompatible();
    nextSession.updateRenderState({ baseLayer: new XRWebGLLayer(nextSession, gl) });
    xrSpace = await nextSession.requestReferenceSpace("local-floor")
      .catch(() => nextSession.requestReferenceSpace("local"));
    quad = makeQuad();
    statusQuad = makeQuad();
    statusTextureText = null;
    xrSession = nextSession;
    nextSession.addEventListener("end", () => {
      xrSession = null; xrSpace = null; gl = null; quad = null;
      statusQuad = null;
      enterButton.textContent = "Enter VR";
    });
    enterButton.textContent = "Exit VR";
    nextSession.requestAnimationFrame(onFrame);
  } catch (error) {
    await nextSession.end();
    throw error;
  }
}

enterButton.addEventListener("click", () => {
  void enterVr().catch((error) => show(`Cannot enter VR: ${error.message}`));
});
resetButton.addEventListener("click", () => send({ type: "reset" }));
connect();
void refreshModelStatus();
function refreshVrSupport() {
  if (!navigator.xr) return;
  void navigator.xr.isSessionSupported("immersive-vr").then((supported) => {
    if (supported) enterButton.disabled = false;
  }).catch((error) => show(`WebXR check failed: ${error.message}`));
}
refreshVrSupport();
window.addEventListener("focus", refreshVrSupport);
navigator.xr?.addEventListener?.("devicechange", refreshVrSupport);
