// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Run with: node cosmos3_h_dreams/tests/test_headset_status.cjs
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

let now = 0;
let draws = 0;
const modelStatus = { textContent: "Compiling model…" };
const canvas = { getContext: () => ({ fillRect() {}, fillText() {} }) };
const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
const context = vm.createContext({
  document: {
    getElementById: id => id === "model-status" ? modelStatus
      : { dataset: {}, addEventListener() {} },
    createElement: () => canvas,
  },
  navigator: {}, window: { addEventListener() {} },
  location: { protocol: "https:", host: "localhost" },
  WebSocket: class {}, fetch: () => new Promise(() => {}), setTimeout() {},
  performance: { now: () => now }, HTMLMediaElement: { HAVE_CURRENT_DATA: 2 },
  testGL: new Proxy({ drawArrays: () => draws++ }, {
    get: (target, key) => target[key] || (() => {}),
  }),
  viewer: {
    transform: { matrix: identity },
    views: [0, 1].map(() => ({
      projectionMatrix: identity, transform: { inverse: { matrix: identity } },
    })),
  },
  layer: { getViewport: () => ({ x: 0, y: 0, width: 100, height: 100 }) },
});
vm.runInContext(fs.readFileSync(path.join(__dirname, "../cosmos3_h_dreams/web/headset.js"), "utf8"), context);
vm.runInContext("gl = testGL; quad = {}; statusQuad = {};", context);

function checkFrame(expectedDraws) {
  draws = 0;
  vm.runInContext("renderVideo(viewer, layer)", context);
  assert.equal(draws, expectedDraws);
}

checkFrame(4);
now = 10000;
checkFrame(4); // Loading remains visible in both eyes, even without video.
for (const message of ["Model compiled", "Model ready"]) {
  modelStatus.textContent = message;
  checkFrame(4);
  now += 1999;
  checkFrame(4);
  modelStatus.textContent = message; // Repeated health polls must not restart the timer.
  now += 1;
  checkFrame(2);
  now += 10000;
  checkFrame(2);
}
modelStatus.textContent = "Model startup failed — check server logs";
checkFrame(4);
now += 10000;
checkFrame(4);
console.log("VR success messages hide after two seconds; loading and errors stay visible.");
