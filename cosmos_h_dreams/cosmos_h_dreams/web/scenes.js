// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

"use strict";

const galleryButton = document.getElementById("scene-gallery-button");
const gallery = document.getElementById("scene-gallery");
const sceneGrid = document.getElementById("scene-grid");
const sceneStatus = document.getElementById("scene-status");
const galleryStatus = document.getElementById("scene-gallery-status");
let sceneRequestPending = false;
let sceneHealth = {};

function updateScenes(health) {
  sceneHealth = health;
  const scenes = health.scenes || [];
  if (JSON.stringify(scenes) !== sceneGrid.dataset.scenes) {
    sceneGrid.replaceChildren(...scenes.map((scene) => {
      const card = document.createElement("button");
      card.type = "button";
      card.dataset.name = scene.name;
      const preview = document.createElement("img");
      preview.src = `/scene/frame?name=${encodeURIComponent(scene.name)}`;
      preview.alt = "";
      preview.loading = "lazy";
      const label = document.createElement("span");
      label.textContent = scene.label;
      card.append(preview, label);
      card.addEventListener("click", () => selectScene(scene.name));
      return card;
    }));
    sceneGrid.dataset.scenes = JSON.stringify(scenes);
  }
  galleryButton.disabled = scenes.length === 0;
  for (const card of sceneGrid.children) {
    card.disabled = !health.ready || health.scene_switching || sceneRequestPending;
    card.setAttribute("aria-pressed", String(card.dataset.name === health.active_scene));
  }
}

galleryButton.addEventListener("click", () => gallery.showModal());

async function selectScene(name) {
  if (sceneRequestPending) return;
  sceneRequestPending = true;
  galleryStatus.textContent = "Changing scene…";
  updateScenes(sceneHealth);
  try {
    const response = await fetch("/scene", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
    if (!response.ok) throw new Error(await response.text());
    sceneHealth = { ...sceneHealth, active_scene: (await response.json()).active_scene };
    sceneStatus.textContent = "Scene ready · Hold both controllers still to calibrate.";
    galleryStatus.textContent = "";
    gallery.close();
  } catch (error) {
    galleryStatus.textContent = `Could not change scene: ${error.message}`;
  } finally {
    sceneRequestPending = false;
    updateScenes(sceneHealth);
  }
}
