// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Run with: node cosmos3_h_dreams/tests/test_scene_selector.cjs
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

function element() {
  return {
    dataset: {}, children: [], attributes: {},
    append(...children) { this.children.push(...children); },
    replaceChildren(...children) { this.children = children; },
    addEventListener(event, callback) { this[event] = callback; },
    setAttribute(name, value) { this.attributes[name] = value; },
    showModal() { this.open = true; },
    close() { this.open = false; },
  };
}
const elements = Object.fromEntries([
  "scene-gallery-button", "scene-gallery", "scene-grid", "scene-status", "scene-gallery-status",
].map(id => [id, element()]));
const grid = elements["scene-grid"];
const dialog = elements["scene-gallery"];
let request;
let finish;
const context = vm.createContext({
  document: { getElementById: id => elements[id], createElement: element },
  fetch: (url, options) => {
    request = { url, options };
    return new Promise(resolve => { finish = resolve; });
  },
  health: {
    ready: true, active_scene: "a", scene_switching: false,
    scenes: [{ name: "a", label: "Scene A" }, { name: "b &.png", label: "Scene B" }],
  },
});
vm.runInContext(fs.readFileSync(path.join(__dirname, "../cosmos3_h_dreams/web/scenes.js"), "utf8"), context);
(async () => {
  vm.runInContext("updateScenes(health)", context);
  assert.equal(grid.children.length, 2);
  assert.equal(grid.children[0].attributes["aria-pressed"], "true");
  assert.equal(grid.children[1].children[1].textContent, "Scene B");
  assert.equal(grid.children[1].children[0].src, "/scene/frame?name=b%20%26.png");
  elements["scene-gallery-button"].click();
  assert.equal(dialog.open, true);
  const failed = grid.children[1].click();
  assert.equal(request.url, "/scene");
  assert.deepEqual(JSON.parse(request.options.body), { name: "b &.png" });
  vm.runInContext("updateScenes(health)", context);
  assert.equal(grid.children[1].disabled, true);
  finish({ ok: false, text: async () => "Scene failed" });
  await failed;
  assert.equal(dialog.open, true);
  assert.match(elements["scene-gallery-status"].textContent, /Scene failed/);
  assert.equal(grid.children[1].disabled, false);
  assert.equal(grid.children[0].attributes["aria-pressed"], "true");
  const success = grid.children[1].click();
  finish({ ok: true, json: async () => ({ active_scene: "b &.png" }) });
  await success;
  assert.equal(dialog.open, false);
  assert.equal(grid.children[1].attributes["aria-pressed"], "true");
  vm.runInContext("updateScenes({ ...health, ready: false })", context);
  assert.equal(grid.children[0].disabled, true);
  assert.equal(elements["scene-gallery-button"].disabled, false); // Previews remain accessible while loading.
  console.log("Scene gallery previews, selection, and error recovery pass.");
})().catch(error => { console.error(error); process.exitCode = 1; });
