import { BLOB_LIMIT } from "./budgets.mjs";
import { defaultCpuBudget } from "./pool-plan.mjs";
import { DetailViewer } from "./viewer.mjs";
const $ = (id) => document.getElementById(id);
let retainedBlobBytes = 0,
  savedUrls = [];
let worker,
  writer,
  chunks = [],
  encodedBytes = 0,
  outputHandle,
  outputName,
  resultUrl,
  busy = false;
const viewer = new DetailViewer();
window.zensrViewer = viewer;
let tileDirectory, repairWriter, repairHandle, repairName;
let repairPreference = false;
let repairClock;
function stopRepairClock() {
  clearInterval(repairClock);
  repairClock = null;
}
let inferenceWorkers = [];
function terminateInference() {
  for (const child of inferenceWorkers) child.terminate();
  inferenceWorkers = [];
}
const state = { done: false, stats: null, error: null };
window.zensrState = state;
function status(text) {
  $("status").textContent = text;
}
function controls(running) {
  busy = running;
  $("start").disabled = running || !$("file").files.length;
  $("cancel").hidden = !running;
  for (const id of [
    "file",
    "model",
    "scale",
    "backend",
    "cpu-workers",
    "cpu-memory",
    "repair",
    "repair-only",
  ])
    $(id).disabled = running;
  $("repair-only").disabled = running || !isJpegInput();
  $("repair").disabled =
    running ||
    !(
      $("file").files[0]?.type === "image/jpeg" ||
      /\.jpe?g$/i.test($("file").files[0]?.name || "")
    );
}
function savedList() {
  try {
    return JSON.parse(localStorage.getItem("zensr-outputs") || "[]");
  } catch {
    return [];
  }
}
async function showSaved() {
  if (!navigator.storage?.getDirectory) return;
  const root = await navigator.storage.getDirectory();
  for (const url of savedUrls) URL.revokeObjectURL(url);
  savedUrls = [];
  $("saved").replaceChildren();
  for (const item of savedList()) {
    try {
      const h = await root.getFileHandle(item.name);
      const file = await h.getFile();
      const a = document.createElement("a");
      a.href = URL.createObjectURL(file);
      savedUrls.push(a.href);
      a.download = item.name;
      a.textContent = `Saved locally: ${item.label}`;
      $("saved").append(a);
    } catch {
      /* A browser may evict temporary storage; never pretend the file exists. */
    }
  }
}
async function destination(repairOnly) {
  const stem =
    $("file")
      .files[0].name.replace(/\.[^.]+$/, "")
      .replace(/[^a-zA-Z0-9_-]/g, "_")
      .slice(0, 45) || "image";
  outputName = `${stem}-${repairOnly ? "repair-1x" : $("model").value + "-" + $("scale").value + "x"}-${Date.now()}.png`;
  writer = null;
  outputHandle = null;
  tileDirectory = null;
  chunks = [];
  encodedBytes = 0;
  if (navigator.storage?.getDirectory) await navigator.storage.getDirectory();
}
async function openDestination() {
  if (navigator.storage?.getDirectory) {
    const root = await navigator.storage.getDirectory();
    outputHandle = await root.getFileHandle(outputName, { create: true });
    writer = await outputHandle.createWritable();
    tileDirectory = await root.getDirectoryHandle(outputName + ".tiles", {
      create: true,
    });
  }
}
async function fail(error) {
  stopRepairClock();
  terminateInference();
  worker?.terminate();
  worker = null;
  chunks = [];
  if (repairWriter) {
    try {
      await repairWriter.abort();
    } catch {}
    repairWriter = null;
  }
  if (writer) {
    try {
      await writer.abort();
    } catch {}
    writer = null;
  }
  state.error = String(error);
  status(state.error);
  controls(false);
}
function isJpegInput() {
  const file = $("file").files[0];
  return file?.type === "image/jpeg" || /\.jpe?g$/i.test(file?.name || "");
}
$("file").addEventListener("change", () => {
  $("filename").textContent =
    $("file").files[0]?.name || "JPEG or PNG · up to 12.6 MP";
  $("start").disabled = !$("file").files.length;
  const jpeg =
    $("file").files[0]?.type === "image/jpeg" ||
    /\.jpe?g$/i.test($("file").files[0]?.name || "");
  $("repair").disabled = !jpeg;
  $("repair-only").disabled = !jpeg;
  $("repair").checked = jpeg && repairPreference;
});
$("repair").addEventListener("change", () => {
  repairPreference = $("repair").checked;
});
$("wipe").addEventListener("input", () => {
  const v = $("wipe").value;
  $("after").style.clipPath = `inset(0 0 0 ${v}%)`;
  $("divider").style.left = v + "%";
});
$("cancel").addEventListener("click", () =>
  fail("Cancelled. No completed result was replaced."),
);
$("start").addEventListener("click", () => startJob(false));
$("repair-only").addEventListener("click", () => startJob(true));
async function startJob(repairOnly) {
  if (busy) return;
  controls(true);
  $("cancel").disabled = true;
  state.done = false;
  state.error = null;
  state.stats = null;
  state.decode = null;
  state.repairReport = null;
  state.checkpoint = null;
  $("repair-download").hidden = true;
  $("download").hidden = true;
  $("progress").value = 0;
  $("detail").textContent = "";
  $("download").textContent = repairOnly
    ? "Download repaired PNG (1×) ↓"
    : "Download full-size PNG ↓";
  $("repair-report").textContent =
    repairOnly || $("repair").checked
      ? "Canon4 repair is queued before SR."
      : "JPEG repair is off.";
  try {
    await destination(repairOnly);
    worker = new Worker("./worker.mjs", { type: "module" });
    $("cancel").disabled = false;
    worker.onerror = (e) => {
      if (worker === currentWorker) fail(e.message);
    };
    const currentWorker = worker;
    worker.onmessage = async ({ data: m }) => {
      if (worker !== currentWorker) return;
      try {
        if (m.type === "decode") {
          const child = new Worker("./decode-worker.mjs", { type: "module" }),
            channel = new MessageChannel();
          child.onerror = (e) => {
            if (worker === currentWorker) fail(e.message);
          };
          inferenceWorkers.push(child);
          child.postMessage({ port: channel.port2, jpeg: m.jpeg }, [
            channel.port2,
            m.jpeg.buffer,
          ]);
          currentWorker.postMessage(
            { type: "ack", id: m.id, ports: [channel.port1] },
            [channel.port1],
          );
          return;
        }
        if (m.type === "decode-complete") {
          terminateInference();
          state.decode = m.report;
        }
        if (m.type === "repair") {
          const began = performance.now();
          stopRepairClock();
          repairClock = setInterval(() => {
            if (worker === currentWorker)
              status(
                `JPEG repair · ${Math.floor((performance.now() - began) / 1000)} s · one CPU worker`,
              );
          }, 1000);
          const child = new Worker("./repair-worker.mjs", { type: "module" }),
            channel = new MessageChannel();
          child.onerror = (e) => {
            if (worker === currentWorker) fail(e.message);
          };
          inferenceWorkers.push(child);
          child.postMessage(
            {
              port: channel.port2,
              jpeg: m.jpeg,
              weights: m.weights,
            },
            [channel.port2, m.jpeg.buffer, m.weights.buffer],
          );
          currentWorker.postMessage(
            { type: "ack", id: m.id, ports: [channel.port1] },
            [channel.port1],
          );
          return;
        }
        if (m.type === "repair-complete") {
          stopRepairClock();
          terminateInference();
          state.repairReport = m.report;
          $("repair-report").textContent =
            `JPEG repair ${m.report.applied ? "applied" : "bypassed by automatic routing"} · ${m.report.seconds.toFixed(1)} s · ${m.report.math === "relaxed" ? "relaxed SIMD" : "unfused SIMD fallback"}`;
        }
        if (m.type === "pool") {
          const ports = [];
          for (let i = 0; i < m.count; i++) {
            const child = new Worker("./tile-worker.mjs", { type: "module" }),
              channel = new MessageChannel();
            child.onerror = (e) => {
              if (worker === currentWorker) fail(e.message);
            };
            inferenceWorkers.push(child);
            ports.push(channel.port1);
            child.postMessage({ port: channel.port2, bytes: m.bytes }, [
              channel.port2,
            ]);
          }
          currentWorker.postMessage({ type: "ack", id: m.id, ports }, ports);
          return;
        }
        if (m.type === "checkpoint-start") {
          status("Saving the completed 1× JPEG stage…");
          const root = await navigator.storage.getDirectory();
          repairName = outputName.replace(/\.png$/, "-jpeg-stage.png");
          repairHandle = await root.getFileHandle(repairName, { create: true });
          repairWriter = await repairHandle.createWritable();
        } else if (m.type === "checkpoint-chunk")
          await repairWriter.write(m.data);
        else if (m.type === "checkpoint-done") {
          $("cancel").disabled = true;
          await repairWriter.close();
          repairWriter = null;
          const file = await repairHandle.getFile(),
            url = URL.createObjectURL(file);
          $("repair-download").href = url;
          $("repair-download").download = repairName;
          $("repair-download").hidden = false;
          const saved = savedList();
          saved.push({
            name: repairName,
            label: `JPEG stage ${m.width}×${m.height}`,
          });
          localStorage.setItem("zensr-outputs", JSON.stringify(saved));
          state.checkpoint = { name: repairName, bytes: file.size };
          $("cancel").disabled = false;
          await showSaved();
        }
        if (m.type === "status") status(m.text);
        else if (m.type === "original") {
          const ratio = Math.min(1, 1400 / m.width, 1400 / m.height);
          for (const id of ["before", "after"]) {
            $(id).width = Math.max(1, Math.round(m.width * ratio));
            $(id).height = Math.max(1, Math.round(m.height * ratio));
            $(id)
              .getContext("2d")
              .drawImage(m.bitmap, 0, 0, $(id).width, $(id).height);
          }
          $("empty").hidden = true;
          $("viewer").hidden = false;
          await viewer.input(m.bitmap, m, null);
        } else if (m.type === "input") {
          await openDestination();
          for (const id of ["before", "after"]) {
            $(id).width = m.bitmap.width;
            $(id).height = m.bitmap.height;
          }
          $("before").getContext("2d").drawImage(m.bitmap, 0, 0);
          $("after").getContext("2d").drawImage(m.bitmap, 0, 0);
          m.bitmap.close();
          $("empty").hidden = true;
          $("viewer").hidden = false;
          $("dimensions").textContent =
            `${m.width} × ${m.height} → ${m.outputWidth} × ${m.outputHeight}`;
          viewer.result(m, tileDirectory);
          document.querySelector(".label.before").textContent = "Original";
          document.querySelector(".label.after").textContent = "Output";
        } else if (m.type === "tile") {
          $("after")
            .getContext("2d")
            .drawImage(m.bitmap, m.x, m.y, m.width, m.height);
          m.bitmap.close();
          await viewer.tile(m);
          $("progress").value = m.completed / m.total;
          status(
            `${m.completed} / ${m.total} tiles · ${m.seconds.toFixed(1)} s`,
          );
          $("detail").textContent = m.gpu
            ? `WebGPU · ${(m.gpu.peak / 1048576).toFixed(1)} MiB tracked GPU peak`
            : `WASM CPU · ${m.workers} tile worker${m.workers === 1 ? "" : "s"}`;
        } else if (m.type === "chunk") {
          encodedBytes += m.data.byteLength;
          if (writer) await writer.write(m.data);
          else {
            if (encodedBytes + retainedBlobBytes > BLOB_LIMIT)
              throw new Error(
                "Encoded output exceeds the 128 MiB fallback. Use the HTTPS demo with browser-local file storage.",
              );
            chunks.push(m.data);
          }
        } else if (m.type === "done") {
          $("cancel").disabled = true;
          if (writer) {
            await writer.close();
            writer = null;
            const file = await outputHandle.getFile();
            resultUrl = URL.createObjectURL(file);
            const saved = savedList();
            saved.push({
              name: outputName,
              label: `${m.stats.model.toUpperCase()} ${m.stats.outputWidth}×${m.stats.outputHeight}`,
            });
            localStorage.setItem("zensr-outputs", JSON.stringify(saved));
          } else {
            retainedBlobBytes += encodedBytes;
            resultUrl = URL.createObjectURL(
              new Blob(chunks, { type: "image/png" }),
            );
            chunks = [];
          }
          $("download").href = resultUrl;
          $("download").download = outputName;
          $("download").hidden = false;
          $("detail").textContent =
            `${(encodedBytes / 1048576).toFixed(1)} MiB PNG · ${m.stats.backend.toUpperCase()}`;
          status(
            `Finished in ${(m.stats.processingSeconds ?? m.stats.seconds).toFixed(1)} s`,
          );
          state.done = true;
          state.stats = { ...m.stats, encodedBytes, outputName };
          worker.terminate();
          terminateInference();
          worker = null;
          controls(false);
          await showSaved();
        } else if (m.type === "error") throw new Error(m.error);
        if (m.id !== undefined && worker === currentWorker)
          currentWorker.postMessage({ type: "ack", id: m.id });
      } catch (error) {
        if (worker === currentWorker) await fail(error.message || error);
      }
    };
    worker.postMessage({
      file: $("file").files[0],
      model: $("model").value,
      operation: repairOnly ? "repair" : "sr",
      scale: repairOnly ? 1 : Number($("scale").value),
      backend: $("backend").value,
      cpuWorkers: Number($("cpu-workers").value),
      cpuBudgetMiB: Number($("cpu-memory").value),
      detailTiles: !!navigator.storage?.getDirectory,
      repairEnabled: repairOnly || $("repair").checked,
    });
  } catch (error) {
    await fail(error.message || error);
  }
}
const logical = navigator.hardwareConcurrency || 1;
$("cpu-workers").options[0].textContent = `Auto (${logical} logical CPUs)`;
for (let i = 1; i <= logical; i++)
  $("cpu-workers").add(new Option(String(i), String(i)));
$("cpu-memory").value = String(defaultCpuBudget(navigator.deviceMemory));
$("backend").addEventListener("change", () => {
  $("cpu-options").hidden = $("backend").value !== "wasm";
});
try {
  const adapter = await navigator.gpu?.requestAdapter({
    powerPreference: "high-performance",
  });
  $("backend").value = adapter ? "webgpu" : "wasm";
  $("backend").options[0].disabled = !adapter;
  status(
    adapter
      ? "WebGPU ready. Choose an image."
      : "WASM CPU ready. Choose an image.",
  );
  if (!isSecureContext)
    $("detail").textContent =
      "WebGPU and browser-local file storage require the HTTPS address.";
} catch {
  $("backend").value = "wasm";
  status("WASM CPU ready. Choose an image.");
}
showSaved().catch(() => {});
$("cpu-options").hidden = $("backend").value !== "wasm";

// Gallery comparison uses only same-origin, hash-named retained PNGs.
const galleryQuery = new URLSearchParams(location.search);
if (galleryQuery.has("candidate")) {
  try {
    const paths = ["reference", "candidate"].map((key) =>
      galleryQuery.get(key),
    );
    if (
      paths.some(
        (path) =>
          !/^\/gallery\/(?:images|references)\/[a-f0-9]{64}\.png$/.test(
            path || "",
          ),
      )
    )
      throw new Error("Invalid gallery image path.");
    const blobs = await Promise.all(
      paths.map(async (path) => {
        const response = await fetch(path);
        if (!response.ok) throw new Error("Gallery image is unavailable.");
        const blob = await response.blob();
        if (blob.size > 64 * 1048576)
          throw new Error("Gallery image is too large.");
        return blob;
      }),
    );
    const codec = await import("./codec/zensr_browser_codec.js");
    await codec.default();
    const sizes = await Promise.all(
      blobs.map(async (blob) =>
        codec.dimensions(new Uint8Array(await blob.arrayBuffer())),
      ),
    );
    if (sizes[0].join(",") !== sizes[1].join(","))
      throw new Error("Gallery comparison shapes differ.");
    $("empty").hidden = true;
    $("viewer").hidden = false;
    const [width, height] = sizes[1];
    const candidate = await createImageBitmap(blobs[1]);
    const previewRatio = Math.min(1, 1400 / width, 1400 / height);
    $("after").width = Math.max(1, Math.round(width * previewRatio));
    $("after").height = Math.max(1, Math.round(height * previewRatio));
    $("after")
      .getContext("2d")
      .drawImage(candidate, 0, 0, $("after").width, $("after").height);
    await viewer.input(
      blobs[0],
      { outputWidth: width, outputHeight: height, scale: 1 },
      null,
    );
    viewer.staticResult = candidate;
    viewer.render();
    document.querySelector(".label.before").textContent =
      galleryQuery.get("beforeLabel") || "Reference";
    document.querySelector(".label.after").textContent =
      galleryQuery.get("afterLabel") || "Candidate";
    $("dimensions").textContent = `Gallery comparison · ${width} × ${height}`;
    status("Gallery ready. Zoom, pan or flicker to inspect.");
  } catch (error) {
    status(error.message || String(error));
  }
}
