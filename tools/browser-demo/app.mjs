import { BLOB_LIMIT } from "./budgets.mjs";
import { defaultCpuBudget } from "./pool-plan.mjs";
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
  ])
    $(id).disabled = running;
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
async function destination() {
  const stem =
    $("file")
      .files[0].name.replace(/\.[^.]+$/, "")
      .replace(/[^a-zA-Z0-9_-]/g, "_")
      .slice(0, 45) || "image";
  outputName = `${stem}-${$("model").value}-${$("scale").value}x-${Date.now()}.png`;
  writer = null;
  outputHandle = null;
  chunks = [];
  encodedBytes = 0;
  if (navigator.storage?.getDirectory) {
    const root = await navigator.storage.getDirectory();
    outputHandle = await root.getFileHandle(outputName, { create: true });
    writer = await outputHandle.createWritable();
  }
}
async function fail(error) {
  terminateInference();
  worker?.terminate();
  worker = null;
  chunks = [];
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
$("file").addEventListener("change", () => {
  $("filename").textContent =
    $("file").files[0]?.name || "JPEG or PNG · up to 12.6 MP";
  $("start").disabled = !$("file").files.length;
});
$("wipe").addEventListener("input", () => {
  const v = $("wipe").value;
  $("after").style.clipPath = `inset(0 0 0 ${v}%)`;
  $("divider").style.left = v + "%";
});
$("cancel").addEventListener("click", () =>
  fail("Cancelled. No completed result was replaced."),
);
$("start").addEventListener("click", async () => {
  if (busy) return;
  controls(true);
  $("cancel").disabled = true;
  state.done = false;
  state.error = null;
  state.stats = null;
  $("download").hidden = true;
  $("progress").value = 0;
  $("detail").textContent = "";
  try {
    await destination();
    worker = new Worker("./worker.mjs", { type: "module" });
    $("cancel").disabled = false;
    worker.onerror = (e) => {
      if (worker === currentWorker) fail(e.message);
    };
    const currentWorker = worker;
    worker.onmessage = async ({ data: m }) => {
      if (worker !== currentWorker) return;
      try {
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
        if (m.type === "status") status(m.text);
        else if (m.type === "input") {
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
        } else if (m.type === "tile") {
          $("after")
            .getContext("2d")
            .drawImage(m.bitmap, m.x, m.y, m.width, m.height);
          m.bitmap.close();
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
          status(`Finished in ${m.stats.seconds.toFixed(1)} s`);
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
      scale: Number($("scale").value),
      backend: $("backend").value,
      cpuWorkers: Number($("cpu-workers").value),
      cpuBudgetMiB: Number($("cpu-memory").value),
    });
  } catch (error) {
    await fail(error.message || error);
  }
});
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
