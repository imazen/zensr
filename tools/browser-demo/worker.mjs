import { CORE, HALO, bounds, tileAt, planar, paste } from "./core.mjs";
import { capWasmMemory } from "./budgets.mjs";
import { loadSession, modelBytes } from "./runtime.mjs";
import { planWorkers } from "./pool-plan.mjs";
import { TilePool } from "./pool.mjs";
const wasmBytes = capWasmMemory();
let sequence = 0,
  pending;
const send = async (message, transfer = []) => {
  const id = sequence++;
  const ready = new Promise((resolve) => {
    pending = { id, resolve };
  });
  postMessage({ ...message, id }, transfer);
  return await ready;
};
let running = false;
self.onmessage = (event) => {
  if (event.data.type === "ack") {
    if (pending?.id === event.data.id) {
      pending.resolve(event.data);
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

async function run({
  file,
  model,
  scale,
  backend,
  cpuWorkers = 0,
  cpuBudgetMiB = 512,
}) {
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
  const total = Math.ceil(width / CORE) * Math.ceil(height / CORE);
  const bytes = await modelBytes(model, codec);
  const plan =
    backend === "wasm"
      ? planWorkers({
          logical: navigator.hardwareConcurrency || 1,
          requested: cpuWorkers,
          budgetMiB: cpuBudgetMiB,
          tiles: total,
        })
      : { count: 1 };
  let pool, runtime;
  if (backend === "wasm") {
    const reply = await send({ type: "pool", count: plan.count, bytes });
    pool = new TilePool(reply.ports);
    await pool.ready();
  } else {
    runtime = await loadSession(bytes, backend);
  }
  const gpuStats = () => runtime?.gpuStats() ?? null;
  const memoryBytes = () => (pool ? pool.memoryBytes() : wasmBytes());
  const infer = async (tile) => {
    const rgba = context.getImageData(
      tile.left,
      tile.top,
      tile.ew,
      tile.eh,
    ).data;
    const data = planar(rgba, tile.ew, tile.eh);
    if (pool) return pool.run(tile.index, data, tile.ew, tile.eh);
    const { ort, session } = runtime;
    const input = new ort.Tensor("float32", data, [1, 3, tile.eh, tile.ew]);
    const result = await session.run({ [session.inputNames[0]]: input });
    const tensor = result[session.outputNames[0]];
    if (tensor.dims.join(",") !== [1, 3, tile.eh * 4, tile.ew * 4].join(","))
      throw new Error("Model output shape differs.");
    const floats = tensor.data.slice();
    Object.values(result).forEach((t) => t.dispose());
    input.dispose();
    return floats;
  };
  const halo = HALO[model];
  if (!halo) throw new Error("Unknown model.");
  const encoder = new codec.PngRows(output.width, output.height);
  let completed = 0,
    maxBandBytes = 0,
    maxTileFloats = 0;
  const started = performance.now();
  async function drain(data) {
    if (data.byteLength) await send({ type: "chunk", data }, [data.buffer]);
  }
  await drain(encoder.drain());
  const queue = [];
  let tail = 0;
  const enqueue = () => {
    const index = tail++,
      columns = Math.ceil(width / CORE);
    const tile = {
      ...tileAt(
        (index % columns) * CORE,
        Math.floor(index / columns) * CORE,
        width,
        height,
        halo,
      ),
      index,
    };
    const promise = infer(tile);
    promise.catch(() => {});
    queue.push(promise);
  };
  for (let i = 0; i < plan.count; i++) enqueue();
  try {
    for (let y = 0; y < height; y += CORE) {
      const band = new Uint8Array(
        output.width * Math.min(CORE, height - y) * scale * 3,
      );
      maxBandBytes = Math.max(maxBandBytes, band.byteLength);
      for (let x = 0; x < width; x += CORE) {
        const tile = tileAt(x, y, width, height, halo);
        const floats = await queue.shift();
        if (tail < total) enqueue();
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
            wasmBytes: memoryBytes(),
            workers: plan.count,
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
      workerReservationMiB: plan.reservedMiB ?? 0,
      peakInFlight: pool?.peakInFlight ?? 1,
      gpu: gpuStats(),
      wasmBytes: memoryBytes(),
      workers: plan.count,
    };
    if (runtime) {
      await runtime.session.release();
      runtime.device?.destroy();
    }
    pool?.close();
    postMessage({ type: "done", stats });
  } finally {
    canvas.width = 1;
    canvas.height = 1;
  }
}
