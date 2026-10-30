// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import assert from 'node:assert/strict'
import test from 'node:test'
import { TokenStreamSession } from '../cosmos3_h_dreams/web/token_stream/session.js'
import { RenderLoop } from '../cosmos3_h_dreams/web/token_stream/render/render_loop.js'

test('default preroll presents one chunk without waiting for a second', () => {
  const session = new TokenStreamSession({url: 'ws://localhost/ws'})
  assert.equal(session._preRoll, 1)
  const shown = [], acknowledged = []
  const loop = new RenderLoop({
    paced: true, fps: 10, preRoll: session._preRoll,
    onChunkPresented: id => acknowledged.push(id),
  })
  loop._blit = frame => shown.push(frame)
  loop.enqueue({chunkId: 7, frames: [0, 1, 2, 3]})
  loop._tickPaced(0)
  assert.deepEqual(shown, [0])
  assert.deepEqual(acknowledged, [])
  for (const time of [100, 200, 300]) loop._tickPaced(time)
  assert.deepEqual(shown, [0, 1, 2, 3])
  assert.deepEqual(acknowledged, [7])
  assert.equal(loop._frameQueue.length, 0)
})
