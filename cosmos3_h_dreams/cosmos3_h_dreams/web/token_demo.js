// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { TokenStreamSession } from "./token_stream/session.js"
import { detectWebGPU } from "./token_stream/webgpu.js"

const el = (id) => document.getElementById(id)
let token = null, socket = null, connecting = false
const keys = new Set()
const supportedKeys = new Set(["w","a","s","d","q","e","z","x","arrowup","arrowdown","arrowleft","arrowright","pageup","pagedown"])
function sendKeys() {
  const payload = {type:"keys", keys:[...keys]}
  token?.sendControl(payload)
  if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify(payload))
}
window.addEventListener("keydown", (event) => {
  const key = event.key.toLowerCase()
  if (!supportedKeys.has(key) || el("stop").disabled || ["SELECT","INPUT"].includes(event.target.tagName)) return
  event.preventDefault(); keys.add(key); sendKeys()
})
window.addEventListener("keyup", (event) => { keys.delete(event.key.toLowerCase()); sendKeys() })
window.addEventListener("blur", () => { keys.clear(); sendKeys() })
document.addEventListener("visibilitychange", () => { if (document.hidden) { keys.clear(); sendKeys() } })
setInterval(sendKeys, 200)
const log = (text) => {
  el("log").textContent = `[${new Date().toLocaleTimeString()}] ${text}\n` + el("log").textContent.slice(0, 12000)
}
function stop() {
  keys.clear(); sendKeys()
  token?.stop(); token = null
  socket?.close(); socket = null
  el("pixel").removeAttribute("src")
  el("stop").disabled = true
  el("mode").disabled = el("scene").disabled = false
  el("status").textContent = "Stopped · reconnect to restart from the initial scene"
}
el("stop").onclick = stop
window.addEventListener("pagehide", stop)
el("connect").onclick = async () => {
  if (connecting) return
  connecting = true
  el("connect").disabled = true
  try {
    const mode = el("mode").value
    const gpu = mode === "raw_tokens" ? await detectWebGPU() : null
    if (mode === "raw_tokens" && !gpu) throw new Error("Raw Tokens needs WebGPU. Use Chrome/Edge through localhost SSH forwarding.")
    const response = await fetch("/scene", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({name:el("scene").value})})
    if (!response.ok) throw new Error(await response.text())
    const url = new URL(`/ws?mode=${mode}`, location.href)
    url.protocol = location.protocol === "https:" ? "wss:" : "ws:"
    el("canvas").hidden = mode !== "raw_tokens"
    el("pixel").hidden = mode === "raw_tokens"
    el("mode").disabled = el("scene").disabled = true
    el("stop").disabled = false
    if (mode === "raw_tokens") {
      token = new TokenStreamSession({
        url:url.href, device:gpu.device, canvas:el("canvas"), log,
        onStatus: (status) => { el("status").textContent = status },
        onClose: () => { if (token) stop() },
      })
      token.start()
    } else {
      socket = new WebSocket(url)
      socket.onopen = () => { el("pixel").src = `/video?t=${Date.now()}` }
      socket.onerror = () => { log("Pixel connection failed; check server readiness or active session"); stop() }
      socket.onclose = () => { log("Pixel connection closed"); stop() }
    }
    el("status").textContent = mode === "raw_tokens" ? "Preparing browser decoder · first load may take several minutes" : "Pixel stream · MJPEG / server TAEHV decode"
    log(`Selected mode=${mode}`)
  } catch (error) { stop(); log(error.message) }
  finally { connecting = false }
}
async function health() {
  try {
    const response = await fetch("/healthz", {cache:"no-store"})
    const data = await response.json()
    if (data.controls === "replay") el("instructions").textContent = "Live model generation driven by the recorded instrument trajectory. Keyboard motion is disabled in replay mode."
    if (!el("scene").options.length) {
      for (const scene of data.scenes) el("scene").add(new Option(scene.label, scene.name))
      el("pixel").src = `/scene/frame?name=${encodeURIComponent(data.active_scene)}`
    }
    el("connect").disabled = connecting || !data.ready || data.headset_connected
    if (!data.ready) el("status").textContent = data.model_status
  } catch { el("status").textContent = "Server unavailable · check the SSH tunnel"; el("connect").disabled = true }
  setTimeout(health, 2000)
}
health()
