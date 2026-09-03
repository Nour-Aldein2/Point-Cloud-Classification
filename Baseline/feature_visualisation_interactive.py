"""
Interactive latent-space visualisation for ModelNet10 / PointNet.

The t-SNE or UMAP map is displayed as a conventional class-coloured scatter
plot. Clicking a marker opens that sample as a larger, rotatable 3D point cloud
in the "Selected shape" panel. By default, the inspector uses 2,048 points.

ModelNet's Z-up coordinates are rotated into Three.js's Y-up coordinate system
with ``(x, y, z) -> (x, z, -y)``. The minus sign preserves handedness; a plain
Y/Z swap would mirror the shape.

Example
-------
python feature_visualisation_scatter_preview.py \
    --ckpt checkpoints/best_model.pt \
    --split test \
    --method tsne \
    --output ../Figures/feature_visualisation_interactive.html

For UMAP, install ``umap-learn`` and use ``--method umap``.

The generated HTML loads Three.js from a pinned CDN URL. Serve the output
directory rather than opening the file through ``file://``:

    cd ../Figures
    python -m http.server 8000

Then open http://localhost:8000/feature_space_tsne_baseline.html.
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from data_utils import PointCloudData
from model import PointNetBaseline


PALETTES = {
    "house_scape": [
        "#5d9781",
        "#ab3d66",
        "#73b0c9",
        "#7f6f8c",
        "#cad8c9",
        "#2b6e72",
        "#75bcc1",
        "#d6a65c",
        "#b86b4b",
        "#566b8f",
    ],
}

# Pinned so a future Three.js release cannot silently change the generated page.
THREE_VERSION = "0.185.1"


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>__DOCUMENT_TITLE__</title>
  <style>
    :root {
      color-scheme: light;
      --paper: #f6f5f1;
      --panel: rgba(255, 255, 255, 0.95);
      --ink: #252a2d;
      --muted: #697277;
      --line: #d9dcda;
      --accent: #2b6e72;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
                   "Segoe UI", sans-serif;
    }

    * { box-sizing: border-box; }
    html, body { width: 100%; height: 100%; margin: 0; overflow: hidden; }
    body { background: var(--paper); color: var(--ink); }

    #app {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 320px;
      width: 100%;
      height: 100%;
    }

    #stage {
      position: relative;
      min-width: 0;
      min-height: 0;
      overflow: hidden;
      background:
        radial-gradient(circle at 25% 15%, rgba(255,255,255,0.96), transparent 35%),
        var(--paper);
    }

    #latent-canvas { display: block; width: 100%; height: 100%; outline: none; }

    #title-card {
      position: absolute;
      top: 18px;
      left: 20px;
      max-width: min(660px, calc(100% - 40px));
      padding: 12px 15px;
      border: 1px solid rgba(217, 220, 218, 0.8);
      border-radius: 12px;
      background: rgba(255, 255, 255, 0.88);
      box-shadow: 0 8px 30px rgba(35, 42, 45, 0.08);
      backdrop-filter: blur(8px);
      pointer-events: none;
    }

    #title-card h1 { margin: 0 0 4px; font-size: 18px; line-height: 1.25; }
    #title-card p { margin: 0; color: var(--muted); font-size: 12px; line-height: 1.45; }

    #tooltip {
      position: absolute;
      display: none;
      z-index: 10;
      min-width: 165px;
      padding: 8px 10px;
      border: 1px solid rgba(0,0,0,0.08);
      border-radius: 8px;
      background: rgba(27, 31, 33, 0.92);
      color: white;
      box-shadow: 0 8px 20px rgba(0,0,0,0.18);
      font-size: 12px;
      line-height: 1.4;
      pointer-events: none;
      transform: translate(12px, 12px);
    }

    #tooltip .muted { color: rgba(255,255,255,0.72); }

    aside {
      min-width: 0;
      overflow-y: auto;
      border-left: 1px solid var(--line);
      background: var(--panel);
      padding: 18px;
      box-shadow: -8px 0 26px rgba(32, 38, 40, 0.04);
    }

    #inspector {
      position: relative;
      min-width: 260px;
    }

    #inspector-resize-edge {
      position: absolute;
      z-index: 30;
      top: 0;
      bottom: 0;
      left: -5px;
      width: 10px;
      cursor: ew-resize;
      touch-action: none;
    }

    #inspector-resize-edge:hover,
    #inspector-resize-edge.dragging {
      background: rgba(43, 110, 114, 0.10);
    }

    section + section { margin-top: 20px; }
    h2 { margin: 0 0 10px; font-size: 13px; letter-spacing: 0.02em; }
    .small { color: var(--muted); font-size: 11px; line-height: 1.45; }

    .control-row {
      display: grid;
      grid-template-columns: 96px minmax(0, 1fr) 42px;
      align-items: center;
      gap: 8px;
      margin: 10px 0;
      font-size: 12px;
    }

    input[type="range"] { width: 100%; accent-color: var(--accent); }

    button {
      appearance: none;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: white;
      color: var(--ink);
      padding: 7px 10px;
      font: inherit;
      font-size: 12px;
      cursor: pointer;
    }

    button:hover { border-color: #aab2ae; background: #fbfbfa; }
    .button-row { display: flex; flex-wrap: wrap; gap: 7px; }

    #legend { display: grid; gap: 7px; }
    .legend-item {
      display: grid;
      grid-template-columns: 18px 12px minmax(0, 1fr) auto;
      align-items: center;
      gap: 7px;
      font-size: 12px;
    }
    .legend-item input { margin: 0; }
    .swatch { width: 11px; height: 11px; border-radius: 50%; }
    .count { color: var(--muted); font-variant-numeric: tabular-nums; }

    #preview-wrap {
      position: relative;
      width: 100%;
      aspect-ratio: 1 / 0.82;
      overflow: hidden;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: #f0f1ee;
    }

    #preview-resize-edge {
      position: absolute;
      z-index: 20;
      left: 0;
      right: 0;
      bottom: 0;
      height: 10px;
      cursor: ns-resize;
      touch-action: none;
    }

    #preview-resize-edge:hover,
    #preview-resize-edge.dragging {
      background: rgba(43, 110, 114, 0.12);
    }
    #preview-canvas { display: block; width: 100%; height: 100%; }
    #preview-empty {
      position: absolute;
      inset: 0;
      display: grid;
      place-items: center;
      padding: 24px;
      color: var(--muted);
      text-align: center;
      font-size: 12px;
      pointer-events: none;
    }
    #selection-info { margin-top: 9px; min-height: 55px; }
    #selection-info strong { display: block; margin-bottom: 3px; font-size: 13px; }

    .status {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      margin-top: 8px;
      color: var(--muted);
      font-size: 11px;
    }
    .status::before {
      content: "";
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background: #5d9781;
    }

    @media (max-width: 760px) {
      #app { grid-template-columns: 1fr; grid-template-rows: minmax(55%, 1fr) auto; }
      aside {
        max-height: 45vh;
        border-left: 0;
        border-top: 1px solid var(--line);
      }
      #preview-wrap { max-width: 290px; }
      #inspector-resize-edge { display: none; }
    }
  </style>

  <script type="importmap">
  {
    "imports": {
      "three": "https://cdn.jsdelivr.net/npm/three@__THREE_VERSION__/build/three.module.js",
      "three/addons/": "https://cdn.jsdelivr.net/npm/three@__THREE_VERSION__/examples/jsm/"
    }
  }
  </script>
</head>
<body>
  <div id="app">
    <main id="stage">
      <canvas id="latent-canvas" aria-label="Interactive latent-space scatter plot"></canvas>
      <div id="title-card">
        <h1 id="page-title"></h1>
        <p id="page-subtitle"></p>
      </div>
      <div id="tooltip"></div>
    </main>

    <aside id="inspector">
      <div id="inspector-resize-edge" aria-hidden="true" title="Drag to resize inspector"></div>
      <section>
        <h2>View</h2>
        <div class="control-row">
          <label for="marker-size">Marker size</label>
          <input id="marker-size" type="range" min="8" max="25" step="1" value="14" />
          <output id="marker-size-value">14</output>
        </div>
        <div class="control-row">
          <label for="preview-point-size">Preview dots</label>
          <input id="preview-point-size" type="range" min="0.010" max="0.100" step="0.005" value="0.050" />
          <output id="preview-point-size-value">0.050</output>
        </div>
        <div class="button-row">
          <button id="reset-view" type="button">Reset view</button>
          <button id="show-all" type="button">Show all</button>
          <button id="hide-all" type="button">Hide all</button>
        </div>
        <p class="small">Drag to pan, scroll or pinch to zoom, hover for metadata, and click a point to inspect its 3D shape.</p>
      </section>

      <section>
        <h2>Classes</h2>
        <div id="legend"></div>
      </section>

      <section>
        <h2>Selected shape</h2>
        <div id="preview-wrap">
          <canvas id="preview-canvas" aria-label="Rotatable selected point cloud"></canvas>
          <div id="preview-empty">Click a point in the latent scatter plot.</div>
          <div id="preview-resize-edge" aria-hidden="true" title="Drag to resize preview"></div>
        </div>
        <div id="selection-info" class="small"></div>
        <p class="small">Drag the selected shape to rotate it; scroll or pinch to zoom. Drag the preview's bottom edge to resize it.</p>
      </section>

      <section>
        <h2>Data</h2>
        <div id="dataset-summary" class="small"></div>
        <div class="status">WebGL renderer active</div>
      </section>
    </aside>
  </div>

  <script id="latent-data" type="application/json">__PAYLOAD__</script>

  <script type="module">
    import * as THREE from "three";
    import { MapControls } from "three/addons/controls/MapControls.js";
    import { OrbitControls } from "three/addons/controls/OrbitControls.js";

    const data = JSON.parse(document.getElementById("latent-data").textContent);

    function decodeBase64(base64, TypedArrayConstructor) {
      const binary = atob(base64);
      const bytes = new Uint8Array(binary.length);
      const chunk = 1 << 16;
      for (let start = 0; start < binary.length; start += chunk) {
        const end = Math.min(start + chunk, binary.length);
        for (let i = start; i < end; i += 1) bytes[i] = binary.charCodeAt(i);
      }
      return new TypedArrayConstructor(bytes.buffer);
    }

    const projection = decodeBase64(data.arrays.projection, Float32Array);
    const labels = decodeBase64(data.arrays.labels, Uint16Array);
    const sampleIds = decodeBase64(data.arrays.sample_ids, Uint32Array);
    let quantisedClouds = null;

    function getQuantisedClouds() {
      if (quantisedClouds === null) {
        quantisedClouds = decodeBase64(data.arrays.point_clouds, Int16Array);
      }
      return quantisedClouds;
    }

    const nSamples = data.n_samples;
    const pointsPerShape = data.points_per_shape;
    const classVisible = data.classes.map(() => true);
    const classCounts = data.classes.map(() => 0);
    for (let i = 0; i < nSamples; i += 1) classCounts[labels[i]] += 1;

    document.getElementById("page-title").textContent = data.title;
    document.getElementById("page-subtitle").textContent =
      `${nSamples.toLocaleString()} samples are visualised by projecting the model's learned embeddings ` +
      `into two dimensions with ${data.method_name}, with classes colour-coded. ` +
      `The projection highlights class-dependent clustering in the learned feature space. ` +
      `Click any point to inspect the corresponding 3D shape.`;
    document.getElementById("dataset-summary").textContent =
      `${data.split_name} split; ${data.total_samples.toLocaleString()} embeddings reduced, ` +
      `${nSamples.toLocaleString()} included in this page. ${data.orientation_label}.`;

    const stage = document.getElementById("stage");
    const canvas = document.getElementById("latent-canvas");
    const tooltip = document.getElementById("tooltip");

    const app = document.getElementById("app");
    const inspector = document.getElementById("inspector");
    const inspectorResizeEdge = document.getElementById("inspector-resize-edge");

    inspectorResizeEdge.addEventListener("pointerdown", (event) => {
      event.preventDefault();
      inspectorResizeEdge.setPointerCapture(event.pointerId);
      inspectorResizeEdge.classList.add("dragging");

      const startX = event.clientX;
      const startWidth = inspector.getBoundingClientRect().width;

      const onMove = (moveEvent) => {
        const width = Math.min(720, Math.max(260, startWidth + startX - moveEvent.clientX));
        app.style.gridTemplateColumns = `minmax(0, 1fr) ${width}px`;
      };

      const onUp = (upEvent) => {
        inspectorResizeEdge.releasePointerCapture(upEvent.pointerId);
        inspectorResizeEdge.classList.remove("dragging");
        inspectorResizeEdge.removeEventListener("pointermove", onMove);
        inspectorResizeEdge.removeEventListener("pointerup", onUp);
        inspectorResizeEdge.removeEventListener("pointercancel", onUp);
      };

      inspectorResizeEdge.addEventListener("pointermove", onMove);
      inspectorResizeEdge.addEventListener("pointerup", onUp);
      inspectorResizeEdge.addEventListener("pointercancel", onUp);
    });

    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setClearColor(0x000000, 0);
    renderer.outputColorSpace = THREE.SRGBColorSpace;

    const scene = new THREE.Scene();
    const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 100);
    camera.position.set(data.bounds.center_x, data.bounds.center_y, 10);
    camera.lookAt(data.bounds.center_x, data.bounds.center_y, 0);

    const controls = new MapControls(camera, canvas);
    controls.enableRotate = false;
    controls.enableDamping = true;
    controls.dampingFactor = 0.10;
    controls.screenSpacePanning = true;
    controls.minZoom = 0.20;
    controls.maxZoom = 80;
    controls.target.set(data.bounds.center_x, data.bounds.center_y, 0);

    let fittedViewHeight = 1;
    let currentMarkerSize = Number(data.marker_size ?? 14);
    let currentPreviewPointSize = Number(data.preview_point_size ?? 0.05);

    const markerVertexShader = `
      uniform float uMarkerSize;
      uniform float uPixelRatio;

      void main() {
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        gl_PointSize = uMarkerSize * uPixelRatio;
      }
    `;

    const markerFragmentShader = `
      uniform vec3 uColour;
      uniform float uOpacity;

      void main() {
        float radius = length(gl_PointCoord - vec2(0.5));
        if (radius > 0.5) discard;
        float edgeFade = 1.0 - smoothstep(0.42, 0.50, radius);
        gl_FragColor = vec4(uColour, uOpacity * edgeFade);
      }
    `;

    const selectionFragmentShader = `
      uniform vec3 uColour;

      void main() {
        float radius = length(gl_PointCoord - vec2(0.5));
        if (radius > 0.50 || radius < 0.34) discard;
        float outerFade = 1.0 - smoothstep(0.46, 0.50, radius);
        float innerFade = smoothstep(0.34, 0.38, radius);
        gl_FragColor = vec4(uColour, outerFade * innerFade);
      }
    `;

    const classObjects = [];
    const classMaterials = [];

    function buildClassGeometry(classIndex) {
      const positions = new Float32Array(classCounts[classIndex] * 3);
      let out = 0;
      for (let sample = 0; sample < nSamples; sample += 1) {
        if (labels[sample] !== classIndex) continue;
        positions[out] = projection[sample * 2];
        positions[out + 1] = projection[sample * 2 + 1];
        positions[out + 2] = 0;
        out += 3;
      }
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
      return geometry;
    }

    for (let classIndex = 0; classIndex < data.classes.length; classIndex += 1) {
      const material = new THREE.ShaderMaterial({
        uniforms: {
          uMarkerSize: { value: currentMarkerSize },
          uPixelRatio: { value: renderer.getPixelRatio() },
          uColour: { value: new THREE.Color(data.classes[classIndex].colour) },
          uOpacity: { value: 0.82 },
        },
        vertexShader: markerVertexShader,
        fragmentShader: markerFragmentShader,
        transparent: true,
        depthTest: false,
        depthWrite: false,
      });
      const points = new THREE.Points(buildClassGeometry(classIndex), material);
      points.frustumCulled = false;
      points.userData.classIndex = classIndex;
      points.renderOrder = classIndex;
      scene.add(points);
      classObjects.push(points);
      classMaterials.push(material);
    }

    const selectionGeometry = new THREE.BufferGeometry();
    selectionGeometry.setAttribute(
      "position",
      new THREE.BufferAttribute(new Float32Array(3), 3),
    );
    const selectionMaterial = new THREE.ShaderMaterial({
      uniforms: {
        uMarkerSize: { value: currentMarkerSize + 9 },
        uPixelRatio: { value: renderer.getPixelRatio() },
        uColour: { value: new THREE.Color("#252a2d") },
      },
      vertexShader: markerVertexShader,
      fragmentShader: selectionFragmentShader,
      transparent: true,
      depthTest: false,
      depthWrite: false,
    });
    const selectionPoint = new THREE.Points(selectionGeometry, selectionMaterial);
    selectionPoint.visible = false;
    selectionPoint.frustumCulled = false;
    selectionPoint.renderOrder = 100;
    scene.add(selectionPoint);

    function showSelectionMarker(sampleIndex) {
      const position = selectionGeometry.getAttribute("position").array;
      position[0] = projection[sampleIndex * 2];
      position[1] = projection[sampleIndex * 2 + 1];
      position[2] = 0;
      selectionGeometry.getAttribute("position").needsUpdate = true;
      selectionPoint.visible = true;
    }

    function resizeMain() {
      const width = Math.max(1, stage.clientWidth);
      const height = Math.max(1, stage.clientHeight);
      const aspect = width / height;
      renderer.setSize(width, height, false);

      fittedViewHeight = Math.max(
        data.bounds.height,
        data.bounds.width / aspect,
        1e-3,
      ) * 1.08;

      camera.left = -0.5 * fittedViewHeight * aspect;
      camera.right = 0.5 * fittedViewHeight * aspect;
      camera.top = 0.5 * fittedViewHeight;
      camera.bottom = -0.5 * fittedViewHeight;
      camera.updateProjectionMatrix();

      const pixelRatio = renderer.getPixelRatio();
      for (const material of classMaterials) material.uniforms.uPixelRatio.value = pixelRatio;
      selectionMaterial.uniforms.uPixelRatio.value = pixelRatio;
    }

    function resetView() {
      camera.zoom = 1;
      camera.position.set(data.bounds.center_x, data.bounds.center_y, 10);
      controls.target.set(data.bounds.center_x, data.bounds.center_y, 0);
      camera.updateProjectionMatrix();
      controls.update();
    }

    const legend = document.getElementById("legend");
    data.classes.forEach((classInfo, classIndex) => {
      const row = document.createElement("label");
      row.className = "legend-item";

      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.checked = true;
      checkbox.addEventListener("change", () => {
        classVisible[classIndex] = checkbox.checked;
        classObjects[classIndex].visible = checkbox.checked;
        if (selectedSample >= 0 && labels[selectedSample] === classIndex && !checkbox.checked) {
          clearSelection();
        }
      });

      const swatch = document.createElement("span");
      swatch.className = "swatch";
      swatch.style.background = classInfo.colour;

      const name = document.createElement("span");
      name.textContent = classInfo.display_name;

      const count = document.createElement("span");
      count.className = "count";
      count.textContent = classCounts[classIndex].toLocaleString();

      row.append(checkbox, swatch, name, count);
      legend.append(row);
    });

    function setAllClasses(visible) {
      classVisible.fill(visible);
      classObjects.forEach((object) => { object.visible = visible; });
      legend.querySelectorAll('input[type="checkbox"]').forEach((input) => {
        input.checked = visible;
      });
      if (!visible) clearSelection();
    }

    document.getElementById("reset-view").addEventListener("click", resetView);
    document.getElementById("show-all").addEventListener("click", () => setAllClasses(true));
    document.getElementById("hide-all").addEventListener("click", () => setAllClasses(false));

    const markerSizeInput = document.getElementById("marker-size");
    const markerSizeValue = document.getElementById("marker-size-value");
    markerSizeInput.value = String(currentMarkerSize);
    markerSizeValue.value = currentMarkerSize.toFixed(0);
    markerSizeInput.addEventListener("input", () => {
      currentMarkerSize = Number(markerSizeInput.value);
      markerSizeValue.value = currentMarkerSize.toFixed(0);
      for (const material of classMaterials) {
        material.uniforms.uMarkerSize.value = currentMarkerSize;
      }
      selectionMaterial.uniforms.uMarkerSize.value = currentMarkerSize + 9;
    });

    const previewPointSizeInput = document.getElementById("preview-point-size");
    const previewPointSizeValue = document.getElementById("preview-point-size-value");
    previewPointSizeInput.value = String(currentPreviewPointSize);
    previewPointSizeValue.value = currentPreviewPointSize.toFixed(3);
    previewPointSizeInput.addEventListener("input", () => {
      currentPreviewPointSize = Number(previewPointSizeInput.value);
      previewPointSizeValue.value = currentPreviewPointSize.toFixed(3);
      if (previewObject) previewObject.material.size = currentPreviewPointSize;
    });

    const raycaster = new THREE.Raycaster();
    const latentPlane = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0);
    const worldPointer = new THREE.Vector3();
    const projectedAnchor = new THREE.Vector3();

    function nearestSampleFromEvent(event) {
      const rect = canvas.getBoundingClientRect();
      const mouseX = event.clientX - rect.left;
      const mouseY = event.clientY - rect.top;
      const ndc = new THREE.Vector2(
        (mouseX / rect.width) * 2 - 1,
        -(mouseY / rect.height) * 2 + 1,
      );
      raycaster.setFromCamera(ndc, camera);
      raycaster.ray.intersectPlane(latentPlane, worldPointer);

      let bestIndex = -1;
      let bestDistanceSquared = Infinity;
      for (let sample = 0; sample < nSamples; sample += 1) {
        if (!classVisible[labels[sample]]) continue;
        const dx = projection[sample * 2] - worldPointer.x;
        const dy = projection[sample * 2 + 1] - worldPointer.y;
        const d2 = dx * dx + dy * dy;
        if (d2 < bestDistanceSquared) {
          bestDistanceSquared = d2;
          bestIndex = sample;
        }
      }

      if (bestIndex < 0) return -1;

      projectedAnchor
        .set(projection[bestIndex * 2], projection[bestIndex * 2 + 1], 0)
        .project(camera);
      const anchorScreenX = (projectedAnchor.x * 0.5 + 0.5) * rect.width;
      const anchorScreenY = (-projectedAnchor.y * 0.5 + 0.5) * rect.height;
      const pixelDistance = Math.hypot(anchorScreenX - mouseX, anchorScreenY - mouseY);
      const hitRadius = Math.max(9, currentMarkerSize * 0.75 + 3);
      return pixelDistance <= hitRadius ? bestIndex : -1;
    }

    function tooltipText(sampleIndex) {
      const classInfo = data.classes[labels[sampleIndex]];
      return {
        title: classInfo.display_name,
        detail: `Sample ${sampleIds[sampleIndex]} · ${data.method_name} ` +
                `(${projection[sampleIndex * 2].toFixed(2)}, ${projection[sampleIndex * 2 + 1].toFixed(2)})`,
      };
    }

    let pendingPointerEvent = null;
    let hoverFrame = 0;
    canvas.addEventListener("pointermove", (event) => {
      pendingPointerEvent = event;
      if (hoverFrame) return;
      hoverFrame = requestAnimationFrame(() => {
        hoverFrame = 0;
        const sampleIndex = nearestSampleFromEvent(pendingPointerEvent);
        if (sampleIndex < 0) {
          tooltip.style.display = "none";
          canvas.style.cursor = "grab";
          return;
        }
        const text = tooltipText(sampleIndex);
        tooltip.replaceChildren();
        const title = document.createElement("strong");
        title.textContent = text.title;
        const detail = document.createElement("div");
        detail.className = "muted";
        detail.textContent = text.detail;
        tooltip.append(title, detail);
        tooltip.style.left = `${pendingPointerEvent.clientX - stage.getBoundingClientRect().left}px`;
        tooltip.style.top = `${pendingPointerEvent.clientY - stage.getBoundingClientRect().top}px`;
        tooltip.style.display = "block";
        canvas.style.cursor = "pointer";
      });
    });
    canvas.addEventListener("pointerleave", () => {
      tooltip.style.display = "none";
      canvas.style.cursor = "default";
    });

    let pointerDown = null;
    canvas.addEventListener("pointerdown", (event) => {
      pointerDown = { x: event.clientX, y: event.clientY };
    });
    canvas.addEventListener("pointerup", (event) => {
      if (!pointerDown) return;
      const moved = Math.hypot(event.clientX - pointerDown.x, event.clientY - pointerDown.y);
      pointerDown = null;
      if (moved > 5) return;
      const sampleIndex = nearestSampleFromEvent(event);
      if (sampleIndex >= 0) selectSample(sampleIndex);
    });

    const previewCanvas = document.getElementById("preview-canvas");
    const previewWrap = document.getElementById("preview-wrap");
    const previewEmpty = document.getElementById("preview-empty");
    const selectionInfo = document.getElementById("selection-info");
    const previewResizeEdge = document.getElementById("preview-resize-edge");

    previewResizeEdge.addEventListener("pointerdown", (event) => {
      event.preventDefault();
      event.stopPropagation();
      previewResizeEdge.setPointerCapture(event.pointerId);
      previewResizeEdge.classList.add("dragging");

      const startY = event.clientY;
      const startHeight = previewWrap.getBoundingClientRect().height;
      previewWrap.style.aspectRatio = "auto";

      const onMove = (moveEvent) => {
        const height = Math.min(720, Math.max(160, startHeight + moveEvent.clientY - startY));
        previewWrap.style.height = `${height}px`;
      };

      const onUp = (upEvent) => {
        previewResizeEdge.releasePointerCapture(upEvent.pointerId);
        previewResizeEdge.classList.remove("dragging");
        previewResizeEdge.removeEventListener("pointermove", onMove);
        previewResizeEdge.removeEventListener("pointerup", onUp);
        previewResizeEdge.removeEventListener("pointercancel", onUp);
      };

      previewResizeEdge.addEventListener("pointermove", onMove);
      previewResizeEdge.addEventListener("pointerup", onUp);
      previewResizeEdge.addEventListener("pointercancel", onUp);
    });
    const previewRenderer = new THREE.WebGLRenderer({ canvas: previewCanvas, antialias: true, alpha: true });
    previewRenderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    previewRenderer.setClearColor(0x000000, 0);
    previewRenderer.outputColorSpace = THREE.SRGBColorSpace;

    const previewScene = new THREE.Scene();
    const previewCamera = new THREE.PerspectiveCamera(38, 1, 0.01, 100);
    previewCamera.position.set(2.25, 1.55, 2.25);
    const previewControls = new OrbitControls(previewCamera, previewCanvas);
    previewControls.enableDamping = true;
    previewControls.dampingFactor = 0.08;
    previewControls.target.set(0, 0, 0);
    previewControls.minDistance = 1.15;
    previewControls.maxDistance = 8;
    previewControls.update();

    const previewAxes = new THREE.AxesHelper(1.12);
    previewAxes.material.transparent = true;
    previewAxes.material.opacity = 0.32;
    previewScene.add(previewAxes);

    let previewObject = null;
    let selectedSample = -1;

    function resizePreview() {
      const width = Math.max(1, previewWrap.clientWidth);
      const height = Math.max(1, previewWrap.clientHeight);
      previewRenderer.setSize(width, height, false);
      previewCamera.aspect = width / height;
      previewCamera.updateProjectionMatrix();
    }

    function selectSample(sampleIndex) {
      selectedSample = sampleIndex;
      showSelectionMarker(sampleIndex);

      if (previewObject) {
        previewScene.remove(previewObject);
        previewObject.geometry.dispose();
        previewObject.material.dispose();
      }

      const cloudQ = getQuantisedClouds();
      const positions = new Float32Array(pointsPerShape * 3);
      const start = sampleIndex * pointsPerShape * 3;
      for (let i = 0; i < positions.length; i += 1) {
        positions[i] = cloudQ[start + i] / 32767.0;
      }

      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
      geometry.computeBoundingSphere();
      const material = new THREE.PointsMaterial({
        color: data.classes[labels[sampleIndex]].colour,
        size: currentPreviewPointSize,
        sizeAttenuation: true,
      });
      previewObject = new THREE.Points(geometry, material);
      previewScene.add(previewObject);

      previewCamera.position.set(2.25, 1.55, 2.25);
      previewControls.target.set(0, 0, 0);
      previewControls.update();
      previewEmpty.style.display = "none";

      const classInfo = data.classes[labels[sampleIndex]];
      selectionInfo.replaceChildren();
      const heading = document.createElement("strong");
      heading.textContent = classInfo.display_name;
      const details = document.createElement("span");
      details.textContent = `Sample ${sampleIds[sampleIndex]} · ` +
        `${pointsPerShape.toLocaleString()} points · latent coordinate ` +
        `(${projection[sampleIndex * 2].toFixed(3)}, ${projection[sampleIndex * 2 + 1].toFixed(3)})`;
      selectionInfo.append(heading, details);
    }

    function clearSelection() {
      selectedSample = -1;
      selectionPoint.visible = false;
      if (previewObject) {
        previewScene.remove(previewObject);
        previewObject.geometry.dispose();
        previewObject.material.dispose();
        previewObject = null;
      }
      previewEmpty.style.display = "grid";
      selectionInfo.replaceChildren();
    }

    const mainResizeObserver = new ResizeObserver(() => resizeMain());
    const previewResizeObserver = new ResizeObserver(() => resizePreview());
    mainResizeObserver.observe(stage);
    previewResizeObserver.observe(previewWrap);
    resizeMain();
    resizePreview();
    resetView();

    function animate() {
      requestAnimationFrame(animate);
      controls.update();
      previewControls.update();
      renderer.render(scene, camera);
      previewRenderer.render(previewScene, previewCamera);
    }
    animate();
  </script>
</body>
</html>
"""


def load_trained_model(checkpoint_path: str | Path, cfg: Config) -> PointNetBaseline:
    """Load a PointNet baseline checkpoint on CPU."""
    hist_dict = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    model_state_dict = hist_dict["model_state_dict"]

    model = PointNetBaseline(cfg)
    model.load_state_dict(model_state_dict)
    return model


def _make_loader(dataset, cfg: Config) -> DataLoader:
    """Build a deterministic, non-shuffled loader for visualisation."""
    return DataLoader(
        dataset=dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def prepare_data(cfg: Config, split_name: str, render_num_points: int):
    """Create separate loaders for model inference and high-resolution display.

    The model loader retains ``cfg.data.num_points`` so checkpoint inference is
    unchanged. The render loader asks ``PointCloudData`` to sample 2,048 points
    (or the CLI-selected amount) directly from each source shape, rather than
    duplicating an already downsampled model input.
    """
    model_dataset = PointCloudData(
        root_dir=cfg.data.path,
        num_points=cfg.data.num_points,
        split_name=split_name,
        seed=cfg.data.seed,
    )

    render_num_points = (
        cfg.data.num_points if render_num_points <= 0 else int(render_num_points)
    )
    if render_num_points == cfg.data.num_points:
        render_dataset = model_dataset
    else:
        render_dataset = PointCloudData(
            root_dir=cfg.data.path,
            num_points=render_num_points,
            split_name=split_name,
            seed=cfg.data.seed,
        )

    if render_dataset.classes != model_dataset.classes:
        raise RuntimeError("Model and rendering datasets produced different class mappings.")
    if len(render_dataset) != len(model_dataset):
        raise RuntimeError(
            "Model and rendering datasets contain different numbers of samples: "
            f"{len(model_dataset)} and {len(render_dataset)}."
        )

    return (
        _make_loader(model_dataset, cfg),
        _make_loader(render_dataset, cfg),
        model_dataset.classes,
    )


def _batch_xyz(point_cloud_batch: torch.Tensor) -> torch.Tensor:
    """Return point coordinates as B x N x 3, accepting common layouts."""
    if point_cloud_batch.ndim != 3:
        raise ValueError(
            "Expected batch['point_cloud'] to have three dimensions, "
            f"but received shape {tuple(point_cloud_batch.shape)}."
        )

    # B x N x 3 (or B x N x 6 when normals/features follow XYZ).
    if point_cloud_batch.shape[-1] in (3, 6):
        return point_cloud_batch[..., :3]

    # B x 3 x N (or B x 6 x N).
    if point_cloud_batch.shape[1] in (3, 6):
        return point_cloud_batch[:, :3, :].transpose(1, 2)

    raise ValueError(
        "Could not locate the XYZ coordinate dimension in point-cloud batch "
        f"with shape {tuple(point_cloud_batch.shape)}. Expected BxNx3 or Bx3xN."
    )


def extract_embeddings(model, loader, device: str):
    """Extract penultimate features and labels using the checkpoint input size."""
    embeddings: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []

    # Input to the final classifier is the learned penultimate embedding.
    def save_embedding(module, inputs):
        del module
        embeddings.append(inputs[0].detach().cpu())

    hook = model.fc2.register_forward_pre_hook(save_embedding)

    try:
        with torch.inference_mode():
            model.eval()
            for batch in tqdm(
                loader,
                desc="Extracting embeddings",
                ncols=80,
                leave=False,
                colour="#5d9781",
            ):
                points = batch["point_cloud"].to(device, non_blocking=True)
                y_true = torch.as_tensor(batch["category"])

                model(points)
                labels.append(y_true.detach().cpu().reshape(-1))
    finally:
        hook.remove()

    if not embeddings:
        raise ValueError("No samples found for feature visualisation.")

    embedding_array = torch.cat(embeddings, dim=0).numpy()
    label_array = torch.cat(labels, dim=0).numpy()

    if len(embedding_array) != len(label_array):
        raise RuntimeError(
            "Embedding and label counts do not match: "
            f"{len(embedding_array)} and {len(label_array)}."
        )

    return embedding_array, label_array


def extract_render_clouds(loader):
    """Collect the independently sampled high-resolution XYZ point clouds."""
    point_clouds: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []

    for batch in tqdm(
        loader,
        desc="Loading 3D shapes",
        ncols=80,
        leave=False,
        colour="#73b0c9",
    ):
        point_clouds.append(_batch_xyz(batch["point_cloud"]).detach().cpu())
        labels.append(torch.as_tensor(batch["category"]).detach().cpu().reshape(-1))

    if not point_clouds:
        raise ValueError("No point clouds found for feature visualisation.")

    return (
        torch.cat(point_clouds, dim=0).numpy(),
        torch.cat(labels, dim=0).numpy(),
    )


def reduce_embeddings(embeddings: np.ndarray, method: str, seed: int) -> np.ndarray:
    if len(embeddings) < 2:
        raise ValueError("At least two samples are required for feature visualisation.")

    if method == "tsne":
        perplexity = min(30, len(embeddings) - 1)
        reducer = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=seed,
        )
    elif method == "umap":
        try:
            import umap
        except ImportError as exc:
            raise ImportError(
                "UMAP requires the `umap-learn` package. "
                "Install it with `pip install umap-learn`."
            ) from exc

        reducer = umap.UMAP(n_components=2, random_state=seed)
    else:
        raise ValueError(f"Unknown dimensionality reduction method: {method!r}")

    return np.asarray(reducer.fit_transform(embeddings), dtype=np.float32)


def parse_axis_order(
    axis_order: str,
) -> tuple[tuple[int, int, int], tuple[float, float, float]]:
    """Parse a signed output-axis mapping such as ``x,z,-y``.

    The three tokens describe rendered X, Y, and Z respectively. For example,
    ``x,z,-y`` maps a Z-up source cloud to Three.js's Y-up convention using a
    proper -90 degree rotation about X. Compact unsigned forms such as ``xyz``
    and ``xzy`` remain supported for backwards compatibility. Legacy ``xzy``
    is interpreted as ``x,z,-y`` so it performs a rotation rather than the
    mirrored Y/Z swap used by the first version of this viewer.
    """
    cleaned = axis_order.lower().replace(" ", "")
    if cleaned == "xzy":
        tokens = ["x", "z", "-y"]
    else:
        tokens = cleaned.split(",") if "," in cleaned else list(cleaned)

    if len(tokens) != 3:
        raise ValueError(
            "--axis-order must contain three axes, for example 'x,z,-y' or 'xyz'."
        )

    lookup = {"x": 0, "y": 1, "z": 2}
    indices: list[int] = []
    signs: list[float] = []
    used_axes: list[str] = []

    for token in tokens:
        if not token:
            raise ValueError(
                "--axis-order contains an empty axis; use a value such as 'x,z,-y'."
            )

        sign = 1.0
        if token[0] in "+-":
            sign = -1.0 if token[0] == "-" else 1.0
            token = token[1:]

        if token not in lookup:
            raise ValueError(
                "Each --axis-order token must be x, y, z, -x, -y, or -z. "
                f"Received {axis_order!r}."
            )

        indices.append(lookup[token])
        signs.append(sign)
        used_axes.append(token)

    if set(used_axes) != {"x", "y", "z"}:
        raise ValueError(
            "--axis-order must use each source axis exactly once. "
            f"Received {axis_order!r}."
        )

    # Tokens specify rendered coordinates as signed source coordinates. For
    # row-vector points, columns of this matrix are the rendered axes. Reject
    # reflections so an accidental unsigned axis swap cannot mirror objects.
    transform = np.zeros((3, 3), dtype=np.float64)
    for rendered_axis, (source_axis, sign) in enumerate(zip(indices, signs)):
        transform[source_axis, rendered_axis] = sign
    determinant = float(np.linalg.det(transform))
    if determinant < 0:
        raise ValueError(
            f"Axis map {axis_order!r} mirrors the point cloud. Use a proper "
            "rotation instead; for Z-up ModelNet data use 'x,z,-y'."
        )

    return tuple(indices), tuple(signs)


def canonical_axis_order(axis_order: str) -> str:
    """Return a normalised comma-separated signed axis map."""
    indices, signs = parse_axis_order(axis_order)
    axis_names = "xyz"
    return ",".join(
        ("-" if sign < 0 else "") + axis_names[index]
        for index, sign in zip(indices, signs)
    )


def axis_orientation_label(axis_order: str) -> str:
    """Create a readable description for the generated webpage."""
    indices, signs = parse_axis_order(axis_order)
    axis_names = "XYZ"
    source_axes = [
        ("−" if sign < 0 else "") + axis_names[index]
        for index, sign in zip(indices, signs)
    ]
    return f"Rendered (X, Y, Z) = source ({', '.join(source_axes)})"


def prepare_clouds_for_web(
    point_clouds: np.ndarray,
    points_per_shape: int,
    seed: int,
    axis_order: str,
) -> np.ndarray:
    """Sample, centre, unit-normalise, and orient all point clouds."""
    if point_clouds.ndim != 3 or point_clouds.shape[-1] != 3:
        raise ValueError(
            "point_clouds must have shape (samples, points, 3); "
            f"received {point_clouds.shape}."
        )
    if points_per_shape <= 0:
        points_per_shape = point_clouds.shape[1]

    axes, signs = parse_axis_order(axis_order)
    signs_array = np.asarray(signs, dtype=np.float32)
    rng = np.random.default_rng(seed)
    prepared = np.empty((len(point_clouds), points_per_shape, 3), dtype=np.float32)

    for sample_index, cloud in enumerate(point_clouds):
        cloud = np.asarray(cloud, dtype=np.float32)
        cloud = cloud[np.isfinite(cloud).all(axis=1)]
        if len(cloud) == 0:
            raise ValueError(f"Point cloud {sample_index} has no finite XYZ points.")

        choice = rng.choice(
            len(cloud),
            size=points_per_shape,
            replace=len(cloud) < points_per_shape,
        )
        sampled = cloud[choice]
        sampled = sampled - sampled.mean(axis=0, keepdims=True)
        radius = np.linalg.norm(sampled, axis=1).max()
        if not np.isfinite(radius) or radius < 1e-12:
            radius = 1.0
        sampled = sampled / radius
        prepared[sample_index] = sampled[:, axes] * signs_array

    return np.clip(prepared, -1.0, 1.0)


def choose_samples(labels: np.ndarray, max_shapes: int, seed: int) -> np.ndarray:
    """Choose a deterministic approximately stratified subset for the webpage."""
    labels = np.asarray(labels).reshape(-1)
    n_samples = len(labels)
    if max_shapes <= 0 or max_shapes >= n_samples:
        return np.arange(n_samples, dtype=np.int64)

    rng = np.random.default_rng(seed)
    class_ids, class_counts = np.unique(labels, return_counts=True)
    allocations = np.zeros(len(class_ids), dtype=np.int64)
    desired = max_shapes * class_counts / class_counts.sum()

    # Iterative largest-deficit allocation gives each class an approximately
    # proportional share without depending on sklearn's split constraints.
    for _ in range(max_shapes):
        available = allocations < class_counts
        scores = desired - allocations
        scores[~available] = -np.inf
        # Tiny random jitter only resolves exact ties deterministically by seed.
        scores = scores + rng.random(len(scores)) * 1e-9
        allocations[int(np.argmax(scores))] += 1

    selected: list[np.ndarray] = []
    for class_id, allocation in zip(class_ids, allocations):
        if allocation == 0:
            continue
        class_indices = np.flatnonzero(labels == class_id)
        chosen = rng.choice(class_indices, size=int(allocation), replace=False)
        selected.append(chosen)

    return np.sort(np.concatenate(selected).astype(np.int64))


def _encode_array(array: np.ndarray, dtype: str) -> str:
    contiguous = np.ascontiguousarray(array, dtype=np.dtype(dtype))
    return base64.b64encode(contiguous.tobytes()).decode("ascii")


def _normalise_class_items(classes: dict, colours: Iterable[str]):
    class_items = sorted(classes.items(), key=lambda item: item[1])
    colours = tuple(colours)
    if len(class_items) > len(colours):
        raise ValueError(f"Not enough colours for {len(class_items)} classes.")

    result = []
    for slot, ((class_name, class_id), colour) in enumerate(zip(class_items, colours)):
        result.append(
            {
                "slot": slot,
                "class_id": int(class_id),
                "name": str(class_name),
                "display_name": str(class_name).replace("_", " ").title(),
                "colour": colour,
            }
        )
    return result


def make_feature_plot(
    projection: np.ndarray,
    labels: np.ndarray,
    point_clouds: np.ndarray,
    classes: dict,
    method: str,
    split_name: str,
    colours: tuple[str, ...] = tuple(PALETTES["house_scape"]),
    output_path: str | Path = "./Figures/feature_space_tsne.html",
    points_per_shape: int = 2048,
    max_shapes: int = 0,
    marker_size: float = 14.0,
    preview_point_size: float = 0.05,
    axis_order: str = "x,z,-y",
    seed: int = 0,
    three_version: str = THREE_VERSION,
) -> Path:
    """Write an interactive scatter plot with a rotatable selected-shape preview."""
    projection = np.asarray(projection, dtype=np.float32)
    labels = np.asarray(labels).reshape(-1)
    point_clouds = np.asarray(point_clouds)

    if projection.ndim != 2 or projection.shape[1] != 2:
        raise ValueError(f"projection must have shape (samples, 2); received {projection.shape}.")
    if not np.isfinite(projection).all():
        raise ValueError("projection contains NaN or infinite values.")
    if not (len(projection) == len(labels) == len(point_clouds)):
        raise ValueError("projection, labels, and point_clouds must contain the same samples.")
    if not 8 <= marker_size <= 25:
        raise ValueError("marker_size must be between 8 and 25 pixels.")
    if not 0.01 <= preview_point_size <= 0.10:
        raise ValueError("preview_point_size must be between 0.01 and 0.10.")

    canonical_axis_map = canonical_axis_order(axis_order)

    class_info = _normalise_class_items(classes, colours)
    id_to_slot = {item["class_id"]: item["slot"] for item in class_info}
    try:
        label_slots = np.asarray([id_to_slot[int(label)] for label in labels], dtype=np.uint16)
    except KeyError as exc:
        raise ValueError(f"Label {exc.args[0]} is not present in the classes mapping.") from exc

    sample_indices = choose_samples(labels, max_shapes=max_shapes, seed=seed)
    web_projection = projection[sample_indices]
    web_label_slots = label_slots[sample_indices]
    web_clouds = prepare_clouds_for_web(
        point_clouds[sample_indices],
        points_per_shape=points_per_shape,
        seed=seed,
        axis_order=canonical_axis_map,
    )
    actual_points_per_shape = web_clouds.shape[1]

    min_xy = web_projection.min(axis=0)
    max_xy = web_projection.max(axis=0)
    raw_span = max_xy - min_xy
    dominant_span = max(float(raw_span.max()), 1.0)
    padding = dominant_span * 0.045
    center = (min_xy + max_xy) * 0.5
    width = max(float(raw_span[0] + 2 * padding), dominant_span * 0.20)
    height = max(float(raw_span[1] + 2 * padding), dominant_span * 0.20)

    # Unit-normalised coordinates are quantised to signed 16-bit integers. The
    # point-cloud payload is decoded only after a user first selects a marker.
    quantised_clouds = np.rint(web_clouds * 32767.0).astype(np.int16)

    method_name = method.upper() if method == "umap" else "t-SNE"
    title = f"Learned Feature Space ({method_name}) — PointNet Baseline"
    payload = {
        "title": title,
        "method_name": method_name,
        "split_name": split_name,
        "axis_order": canonical_axis_map,
        "orientation_label": axis_orientation_label(canonical_axis_map),
        "n_samples": int(len(sample_indices)),
        "total_samples": int(len(labels)),
        "points_per_shape": int(actual_points_per_shape),
        "marker_size": float(marker_size),
        "preview_point_size": float(preview_point_size),
        "classes": class_info,
        "bounds": {
            "center_x": float(center[0]),
            "center_y": float(center[1]),
            "width": width,
            "height": height,
        },
        "arrays": {
            "projection": _encode_array(web_projection, "<f4"),
            "labels": _encode_array(web_label_slots, "<u2"),
            "point_clouds": _encode_array(quantised_clouds, "<i2"),
            "sample_ids": _encode_array(sample_indices, "<u4"),
        },
    }

    payload_json = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    html = (
        HTML_TEMPLATE.replace("__DOCUMENT_TITLE__", title)
        .replace("__THREE_VERSION__", three_version)
        .replace("__PAYLOAD__", payload_json)
    )

    output_path = Path(output_path)
    if output_path.suffix.lower() != ".html":
        output_path = output_path.with_suffix(".html")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Saved interactive feature visualisation to: {output_path}")
    print(f"HTML size: {size_mb:.2f} MiB")
    return output_path


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Visualise learned PointNet embeddings as an interactive scatter plot "
            "with a rotatable selected-shape preview."
        )
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        required=True,
        help="Path to the saved checkpoint.",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split to visualise.",
    )
    parser.add_argument(
        "--method",
        choices=("tsne", "umap"),
        default="tsne",
        help="Dimensionality-reduction method.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output HTML path. Defaults to ../Figures/feature_space_<method>_baseline.html.",
    )
    parser.add_argument(
        "--points-per-shape",
        type=int,
        default=2048,
        help=(
            "Number of sampled points stored and rendered for the selected 3D "
            "preview (default: 2048). Use 0 to retain the rendering loader's input count."
        ),
    )
    parser.add_argument(
        "--max-shapes",
        type=int,
        default=0,
        help="Maximum scatter samples embedded in the page; 0 keeps all samples.",
    )
    parser.add_argument(
        "--marker-size",
        type=float,
        default=14.0,
        help="Initial scatter-marker diameter in CSS pixels (range: 8-25; default: 14).",
    )
    parser.add_argument(
        "--preview-point-size",
        type=float,
        default=0.05,
        help="Initial Three.js point size in the selected-shape panel (range: 0.01-0.10; default: 0.05).",
    )
    parser.add_argument(
        "--axis-order",
        "--axis-map",
        dest="axis_order",
        default="x,z,-y",
        help=(
            "Signed mapping from source coordinates to rendered X/Y/Z. "
            "The default 'x,z,-y' rotates ModelNet Z-up clouds into Three.js Y-up "
            "without mirroring them. Use 'xyz' only when your loader already emits Y-up data."
        ),
    )
    parser.add_argument(
        "--three-version",
        default=THREE_VERSION,
        help="Pinned Three.js npm version loaded by the generated webpage.",
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()

    cfg = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_trained_model(args.ckpt, cfg).to(device)
    model_loader, render_loader, classes = prepare_data(
        cfg,
        args.split,
        render_num_points=args.points_per_shape,
    )

    embeddings, labels = extract_embeddings(model, model_loader, device)
    point_clouds, render_labels = extract_render_clouds(render_loader)
    if not np.array_equal(labels, render_labels):
        raise RuntimeError(
            "The high-resolution rendering dataset is not ordered like the model dataset. "
            "Use the same split and deterministic file ordering in PointCloudData."
        )

    projection = reduce_embeddings(embeddings, args.method, cfg.data.seed)

    output_path = args.output
    if output_path is None:
        output_path = Path("../Figures") / f"feature_space_{args.method}_baseline.html"

    make_feature_plot(
        projection=projection,
        labels=labels,
        point_clouds=point_clouds,
        classes=classes,
        method=args.method,
        split_name=args.split,
        output_path=output_path,
        points_per_shape=args.points_per_shape,
        max_shapes=args.max_shapes,
        marker_size=args.marker_size,
        preview_point_size=args.preview_point_size,
        axis_order=args.axis_order,
        seed=cfg.data.seed,
        three_version=args.three_version,
    )


if __name__ == "__main__":
    main()
