// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Orchestrator for the token-streaming path. Given a token-stream URL and a
// WebGPU device, it opens the TokenStreamSocket, reads the session header,
// selects the codec from the registry, assembles per-chunk latent frames
// (buffered by chunk id until the last-in-chunk flag), decodes them with the
// WebGPU VAE decoder, and drives the render loop.
//
// If the VAE decoder cannot start (no WebGPU, an unsupported client, or a
// mismatched artifact), the failure is surfaced and the connection closes.
// It never silently falls back to the pixel path.

import { getDecoder } from "./codec/registry.js"
import { TokenStreamSocket } from "./network/token_socket.js"
import { RenderLoop } from "./render/render_loop.js"
import { VaeDecoder } from "./gpu/vae_decoder.js"

const noop = () => {}

export class TokenStreamSession {
  /**
   * @param {object} options
   * @param {string} options.url token-stream endpoint URL
   * @param {GPUDevice} [options.device] WebGPU device from the capability probe
   * @param {HTMLCanvasElement} [options.canvas] target canvas for presentation
   * @param {RTCDataChannel} [options.controlChannel] control channel for
   *   capability signaling (reserved for negotiation; not required here)
   * @param {(message: string, meta?: object) => void} [options.log] logger
   */
  constructor({ url, device = null, canvas = null, controlChannel = null, log = noop, onStatus = noop, onClose = noop }) {
    this._url = url
    this._device = device
    this._canvas = canvas
    this._controlChannel = controlChannel
    this._log = log
    this._onStatus = onStatus
    this._onClose = onClose

    this._socket = null
    this._decoder = null
    this._vaeDecoder = null
    this._renderLoop = null
    this._sessionHeader = null
    this._vaeReady = false
    this._started = false

    // chunkId -> ordered array of decoded latent frames awaiting completion.
    this._pendingChunks = new Map()

    // Start presentation with the first decoded frame, without waiting for a
    // second latent chunk. Keep decoding all latents to preserve causal state.
    // ?pace=off disables pacing; ?pace=<N> overrides the startup buffer depth.
    const DEFAULT_PRE_ROLL = 1
    const paceParam =
      typeof location !== "undefined"
        ? new URLSearchParams(location.search).get("pace")
        : null
    this._paced = paceParam !== "off" && paceParam !== "0"
    const preRollParam = Number.parseInt(paceParam ?? "", 10)
    this._preRoll = preRollParam > 0 ? preRollParam : DEFAULT_PRE_ROLL
    this._log(
      this._paced
        ? `paced playout (preRoll=${this._preRoll} frames)`
        : "paced playout disabled (blit on arrival)",
      { source: "client" }
    )
  }

  /** Open the socket and begin consuming token frames. Idempotent. */
  start() {
    if (this._started) {
      return
    }
    this._started = true

    this._socket = new TokenStreamSocket(this._url, {
      onOpen: () => this._log("token-stream socket open", { source: "client" }),
      onSessionHeader: (header) => this._onSessionHeader(header),
      onFrame: (frame) => this._onFrame(frame),
      onClose: () => {
        this._log("token-stream socket closed", { source: "client" })
        this.stop()
        this._onClose()
      },
      onError: (error) =>
        this._log(`token-stream error: ${error.message}`, {
          source: "client",
          level: "error",
        }),
    })
    this._socket.open()
  }

  /** Tear down the socket, render loop, and buffered state. */
  stop() {
    this._started = false
    const decoder = this._vaeDecoder
    this._vaeDecoder = null
    if (decoder) {
      Promise.resolve(this._initializing).then(() => decoder.close()).catch(error => {
        this._log(`Decoder cleanup: ${error.message}`)
      })
    }
    if (this._renderLoop) {
      this._renderLoop.stop()
      this._renderLoop = null
    }
    if (this._socket) {
      this._socket.close()
      this._socket = null
    }
    this._pendingChunks.clear()
    this._vaeReady = false
  }

  sendControl(payload) {
    this._socket?.sendControl(payload)
  }

  _onSessionHeader(header) {
    this._sessionHeader = header
    const codecId = header?.codec?.id
    this._log(
      `token session header: codec=${codecId} shape=${JSON.stringify(header?.latent_shape)} T=${header?.frames_per_chunk} fps=${header?.fps}`,
      { source: "client" }
    )

    try {
      this._decoder = getDecoder(codecId)
      this._decoder.configure(header?.codec?.static_params ?? {})
    } catch (error) {
      this._log(`token codec unavailable: ${error.message}`, {
        source: "client",
        level: "error",
      })
      return
    }

    this._renderLoop = new RenderLoop({
      canvas: this._canvas,
      paced: this._paced,
      fps: header?.fps,
      preRoll: this._preRoll,
      // Ack on presentation to keep the two-chunk transport window bounded.
      onChunkBuffered: () => {},
      onChunkPresented: (chunkId) => {
        this._socket?.sendAck(chunkId)
        this._onStatus(`Raw tokens · browser TAEHV decode · displayed chunk ${chunkId}`)
      },
    })
    this._renderLoop.start()

    // Seed and generated chunks wait for initialization before decoding.
    this._initializing = this._initVaeDecoder(header)
  }

  async _initVaeDecoder(header) {
    if (!this._device) {
      this._log("no WebGPU device; token frames will assemble without decode", {
        source: "client",
      })
      this._renderLoop?.showMessage("Token decode unavailable: no WebGPU device")
      this._socket?.close()
      return
    }
    this._vaeDecoder = new VaeDecoder()
    try {
      await this._vaeDecoder.init(header, this._device, this._log)
      if (!this._started) return
      this._vaeReady = true
      this._onStatus("Browser decoder ready · waiting for generated frames")
      const cacheMode = this._vaeDecoder.cacheMode
      if (cacheMode === "cpu") {
        // GPU-resident cache unavailable on this client → decode is ~2x slower;
        // surface it on-screen so a slow stream isn't mistaken for "just slow".
        this._renderLoop?.setBadge("SLOW: CPU decode (~2x)")
      }
      this._log(`VAE decoder ready (precision=${this._vaeDecoder.precision})`, {
        source: "client",
      })
    } catch (error) {
      this._vaeReady = false
      this._log(`VAE decoder not available: ${error.message}`, {
        source: "client",
      })
      // Fail loud on-screen; token mode stays selected (never falls back to pixel).
      this._renderLoop?.showMessage(`Token decode unavailable: ${error.message}`)
      this._socket?.close()
    }
  }

  _onFrame(frame) {
    if (!this._decoder) {
      // No usable codec; ack so the server does not stall on flow control.
      if (frame.isLastInChunk) {
        this._socket?.sendAck(frame.chunkId)
      }
      return
    }

    let latent
    try {
      latent = this._decoder.decode(frame.payload, frame.codecParams)
    } catch (error) {
      this._log(`token frame decode failed: ${error.message}`, {
        source: "client",
        level: "error",
      })
      if (frame.isLastInChunk) {
        this._socket?.sendAck(frame.chunkId)
      }
      return
    }

    let frames = this._pendingChunks.get(frame.chunkId)
    if (!frames) {
      frames = []
      this._pendingChunks.set(frame.chunkId, frames)
    }
    frames[frame.frameIdx] = latent

    if (frame.isLastInChunk) {
      this._pendingChunks.delete(frame.chunkId)
      void this._completeChunk(frame.chunkId, frames, frame.isKeyframe)
    }
  }

  async _completeChunk(chunkId, latentFrames, isSeed = false) {
    await this._initializing
    if (!this._started) return
    const codecId = this._sessionHeader?.codec?.id
    let decodedFloats = 0
    for (const latent of latentFrames) {
      decodedFloats += latent?.length ?? 0
    }
    this._log(
      `token chunk ${chunkId}: ${latentFrames.length} frames, ${codecId} decoded floats=${decodedFloats} (VAE decode pending)`,
      { source: "client" }
    )

    if (!this._vaeReady || !this._vaeDecoder || !this._renderLoop) {
      this._socket?.close()
      return
    }

    try {
      const rgb = await this._vaeDecoder.decode(latentFrames)
      if (!this._started) return
      if (isSeed) {
        // Prime the causal cache; the reference image is already on screen.
        this._socket?.sendAck(chunkId)
      } else {
        this._renderLoop.enqueue({ chunkId, frames: rgb.frames ?? [rgb] })
      }
    } catch (error) {
      this._log(`token chunk ${chunkId} decode failed: ${error.message}`, {
        source: "client",
        level: "error",
      })
      // A missing causal-cache update corrupts every following chunk.
      this._renderLoop?.showMessage(`Token decode failed: ${error.message}`)
      this._socket?.close()
    }
  }
}
