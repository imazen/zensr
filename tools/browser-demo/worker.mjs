import { CORE, HALO, bounds, tileAt, planar, paste } from "./core.mjs";
import { capWasmMemory, capGpuBuffers } from "./budgets.mjs";
const wasmBytes = capWasmMemory();
let sequence = 0,
  pending;
const send = async (message, transfer = []) => {
  const id = sequence++;
  const ready = new Promise((resolve) => {
    pending = { id, resolve };
  });
  postMessage({ ...message, id }, transfer);
  await ready;
};
let running = false;
self.onmessage = (event) => {
  if (event.data.type === "ack") {
    if (pending?.id === event.data.id) {
      pending.resolve();
      pending = undefined;
    }
    return;
  }
  if (running) return;
  running = true;
  run(event.data).catch((error) =>
    postMessage({ type: "error", error: String(error.message || error) }),
  );
};
async function loadSession(model, backend, codec) {
  const manifest = await (await fetch("./models.json")).json();
  const entry = manifest[model];
  if (!entry) throw new Error("Unknown model.");
  const bytes = new Uint8Array(await (await fetch(entry.url)).arrayBuffer());
  if (codec.sha256(bytes) !== entry.sha256)
    throw new Error("Model bytes failed verification.");
  const ort = await import(
    backend === "webgpu"
      ? "./vendor/ort.webgpu.bundle.min.mjs"
      : "./vendor/ort.wasm.bundle.min.mjs"
  );
  ort.env.wasm.wasmPaths = new URL("./vendor/", self.location.href).href;
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.proxy = false;
  let gpuStats = () => null,
    device;
  if (backend === "webgpu") {
    if (!self.isSecureContext || !navigator.gpu)
      throw new Error("WebGPU requires HTTPS and a supported browser.");
    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: "high-performance",
    });
    if (!adapter)
      throw new Error("No WebGPU adapter is available. Choose WASM CPU.");
    // This packaged runtime creates a device from the adapter itself.
    // Instrument its actual request rather than an unused caller device.
    const request = GPUAdapter.prototype.requestDevice;
    GPUAdapter.prototype.requestDevice = async function (descriptor) {
      if (device)
        throw new Error("A second GPU device exceeds the demo budget.");
      device = await request.call(this, descriptor);
      gpuStats = capGpuBuffers(device);
      return device;
    };
    ort.env.webgpu.adapter = adapter;
    postMessage({
      type: "adapter",
      vendor: adapter.info?.vendor,
      architecture: adapter.info?.architecture,
    });
  }
  const session = await ort.InferenceSession.create(bytes, {
    executionProviders: [backend],
    graphOptimizationLevel: "all",
    enableCpuMemArena: false,
    enableMemPattern: false,
    executionMode: "sequential",
    intraOpNumThreads: 1,
  });
  if (backend === "webgpu" && (!device || !gpuStats().peak))
    throw new Error(
      "GPU buffer accounting did not observe runtime allocations.",
    );
  return { ort, session, gpuStats, device };
}
async function run({ file, model, scale, backend }) {
  const codec = await import("./codec/zensr_browser_codec.js");
  await codec.default();
  if (file.size > 64 * 1024 * 1024)
    throw new Error("The demo accepts files up to 64 MiB.");
  let encoded = new Uint8Array(await file.arrayBuffer());
  let size;
  if (encoded[0] === 255 && encoded[1] === 216) {
    const jpeg = await import("./jpeg/zensr_browser_jpeg_probe.js");
    await jpeg.default();
    size = jpeg.dimensions(encoded);
  } else {
    size = codec.dimensions(encoded);
  }
  bounds(size[0], size[1], scale);
  encoded = null;
  const bitmap = await createImageBitmap(file);
  const width = bitmap.width,
    height = bitmap.height;
  const output = bounds(width, height, scale);
  const canvas = new OffscreenCanvas(width, height);
  const context = canvas.getContext("2d", {
    willReadFrequently: true,
    colorSpace: "srgb",
  });
  context.fillStyle = "#fff";
  context.fillRect(0, 0, width, height);
  context.drawImage(bitmap, 0, 0);
  bitmap.close();
  const ratio = Math.min(1, 1400 / width, 1400 / height);
  const preview = await createImageBitmap(canvas, {
    resizeWidth: Math.max(1, Math.round(width * ratio)),
    resizeHeight: Math.max(1, Math.round(height * ratio)),
  });
  await send(
    {
      type: "input",
      bitmap: preview,
      width,
      height,
      outputWidth: output.width,
      outputHeight: output.height,
    },
    [preview],
  );
  postMessage({ type: "status", text: "Loading model…" });
  const runtime = await loadSession(model, backend, codec);
  const { ort, session, gpuStats, device } = runtime;
  const halo = HALO[model];
  if (!halo) throw new Error("Unknown model.");
  const encoder = new codec.PngRows(output.width, output.height);
  let completed = 0,
    total = Math.ceil(width / CORE) * Math.ceil(height / CORE),
    maxBandBytes = 0,
    maxTileFloats = 0;
  const started = performance.now();
  async function drain(data) {
    if (data.byteLength) await send({ type: "chunk", data }, [data.buffer]);
  }
  await drain(encoder.drain());
  try {
    for (let y = 0; y < height; y += CORE) {
      const band = new Uint8Array(
        output.width * Math.min(CORE, height - y) * scale * 3,
      );
      maxBandBytes = Math.max(maxBandBytes, band.byteLength);
      for (let x = 0; x < width; x += CORE) {
        const tile = tileAt(x, y, width, height, halo);
        const rgba = context.getImageData(
          tile.left,
          tile.top,
          tile.ew,
          tile.eh,
        ).data;
        const input = new ort.Tensor(
          "float32",
          planar(rgba, tile.ew, tile.eh),
          [1, 3, tile.eh, tile.ew],
        );
        const result = await session.run({ [session.inputNames[0]]: input });
        const tensor = result[session.outputNames[0]];
        if (
          tensor.dims.join(",") !== [1, 3, tile.eh * 4, tile.ew * 4].join(",")
        )
          throw new Error("Model output shape differs.");
        const floats = tensor.data;
        maxTileFloats = Math.max(maxTileFloats, floats.byteLength);
        const rgb = codec.pack_tile(
          floats,
          tile.ew * 4,
          tile.eh * 4,
          (x - tile.left) * 4,
          (y - tile.top) * 4,
          tile.cw * 4,
          tile.ch * 4,
          scale,
        );
        paste(rgb, band, tile, width, scale);
        for (const t of Object.values(result)) t.dispose();
        input.dispose();
        const display = new Uint8ClampedArray(
          tile.cw * tile.ch * scale * scale * 4,
        );
        for (let i = 0; i < rgb.length / 3; i++) {
          display[i * 4] = rgb[i * 3];
          display[i * 4 + 1] = rgb[i * 3 + 1];
          display[i * 4 + 2] = rgb[i * 3 + 2];
          display[i * 4 + 3] = 255;
        }
        const image = await createImageBitmap(
          new ImageData(display, tile.cw * scale, tile.ch * scale),
          {
            resizeWidth: Math.max(1, Math.ceil(tile.cw * ratio)),
            resizeHeight: Math.max(1, Math.ceil(tile.ch * ratio)),
          },
        );
        completed++;
        await send(
          {
            type: "tile",
            bitmap: image,
            x: x * ratio,
            y: y * ratio,
            width: tile.cw * ratio,
            height: tile.ch * ratio,
            completed,
            total,
            seconds: (performance.now() - started) / 1000,
            gpu: gpuStats(),
            wasmBytes: wasmBytes(),
          },
          [image],
        );
      }
      encoder.push(band, Math.min(CORE, height - y) * scale);
      await drain(encoder.drain());
    }
    await drain(encoder.finish());
    const stats = {
      backend,
      model,
      scale,
      width,
      height,
      outputWidth: output.width,
      outputHeight: output.height,
      tiles: completed,
      seconds: (performance.now() - started) / 1000,
      maxBandBytes,
      maxTileFloats,
      gpu: gpuStats(),
      wasmBytes: wasmBytes(),
    };
    await session.release();
    device?.destroy();
    postMessage({ type: "done", stats });
  } finally {
    canvas.width = 1;
    canvas.height = 1;
  }
}
