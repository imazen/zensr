import {
  CORE,
  HALO,
  bounds,
  tileAt,
  planar,
  paste,
  rgbaFromRgb,
  previewSize,
} from "./core.mjs";
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
  operation = "sr",
  model,
  scale,
  backend,
  cpuWorkers = 0,
  cpuBudgetMiB = 512,
  detailTiles = false,
  repairEnabled = false,
}) {
  const jobStarted = performance.now();
  if (operation === "repair") {
    if (scale !== 1 || !repairEnabled)
      throw new Error("Repair-only requires 1× output and JPEG repair.");
  } else if (operation !== "sr" || ![2, 4].includes(scale)) {
    throw new Error("Choose repair-only or 2×/4× super-resolution.");
  }
  const codec = await import("./codec/zensr_browser_codec.js");
  await codec.default();
  if (file.size > 64 * 1024 * 1024)
    throw new Error("The demo accepts files up to 64 MiB.");
  let encoded = new Uint8Array(await file.arrayBuffer());
  let size;
  const isJpeg = encoded[0] === 255 && encoded[1] === 216;
  if (isJpeg) {
    const jpeg = await import("./jpeg/zensr_browser_jpeg_probe.js");
    await jpeg.default();
    size = jpeg.dimensions(encoded);
  } else {
    size = codec.dimensions(encoded);
  }
  bounds(size[0], size[1], scale);
  let originalCanvas;
  let decodeReport = null;
  if (isJpeg) {
    postMessage({ type: "status", text: "Decoding JPEG with zenjpeg…" });
    const began = performance.now();
    const reply = await send({ type: "decode", jpeg: encoded }, [
      encoded.buffer,
    ]);
    const decoded = await new Promise((resolve, reject) => {
      const port = reply.ports[0];
      port.onmessage = ({ data }) => {
        port.close();
        data.error ? reject(new Error(data.error)) : resolve(data);
      };
    });
    encoded = decoded.jpeg;
    decodeReport = {
      decoder: "zenjpeg",
      width: decoded.width,
      height: decoded.height,
      wasmBytes: decoded.wasmBytes,
      seconds: (performance.now() - began) / 1000,
    };
    originalCanvas = new OffscreenCanvas(decoded.width, decoded.height);
    originalCanvas
      .getContext("2d", { willReadFrequently: true, colorSpace: "srgb" })
      .putImageData(
        new ImageData(
          rgbaFromRgb(decoded.rgb, decoded.width, decoded.height),
          decoded.width,
          decoded.height,
        ),
        0,
        0,
      );
    await send({ type: "decode-complete", report: decodeReport });
  } else {
    // PNG decoding stays in the browser; JPEG bytes never reach this branch.
    const bitmap = await createImageBitmap(file);
    bounds(bitmap.width, bitmap.height, scale);
    originalCanvas = new OffscreenCanvas(bitmap.width, bitmap.height);
    const ctx = originalCanvas.getContext("2d", {
      willReadFrequently: true,
      colorSpace: "srgb",
    });
    ctx.fillStyle = "#fff";
    ctx.fillRect(0, 0, bitmap.width, bitmap.height);
    ctx.drawImage(bitmap, 0, 0);
    bitmap.close();
  }
  const original = await createImageBitmap(originalCanvas);
  await send(
    {
      type: "original",
      bitmap: original,
      width: originalCanvas.width,
      height: originalCanvas.height,
      outputWidth: originalCanvas.width * scale,
      outputHeight: originalCanvas.height * scale,
      scale,
    },
    [original],
  );
  let repaired,
    repairReport = null;
  if (repairEnabled) {
    originalCanvas.width = 1;
    originalCanvas.height = 1;
    const repairStarted = performance.now();
    if (!isJpeg) throw new Error("JPEG artifact repair requires a JPEG input.");
    postMessage({
      type: "status",
      text: "Repairing JPEG with canon4 auto routing…",
    });
    const weights = await modelBytes("repair", codec);
    const reply = await send({ type: "repair", jpeg: encoded, weights }, [
      encoded.buffer,
      weights.buffer,
    ]);
    repaired = await new Promise((resolve, reject) => {
      const port = reply.ports[0];
      port.onmessage = ({ data }) => {
        port.close();
        data.error ? reject(new Error(data.error)) : resolve(data);
      };
    });
    const { rgb, ...report } = repaired;
    report.seconds = (performance.now() - repairStarted) / 1000;
    repairReport = report;
    await send({ type: "repair-complete", report });
  }
  encoded = null;
  let width, height, canvas, context, repairPixels;
  if (repaired) {
    width = repaired.width;
    height = repaired.height;
    bounds(width, height, scale);
    if (repaired.rgb.byteLength !== width * height * 3)
      throw new Error("Repair RGB shape differs.");
    canvas = new OffscreenCanvas(width, height);
    context = canvas.getContext("2d", {
      willReadFrequently: true,
      colorSpace: "srgb",
    });
    const rgba = new Uint8ClampedArray(width * height * 4);
    for (let i = 0; i < width * height; i++) {
      rgba[i * 4] = repaired.rgb[i * 3];
      rgba[i * 4 + 1] = repaired.rgb[i * 3 + 1];
      rgba[i * 4 + 2] = repaired.rgb[i * 3 + 2];
      rgba[i * 4 + 3] = 255;
    }
    context.putImageData(new ImageData(rgba, width, height), 0, 0);
    if (detailTiles && operation === "sr") {
      await send({ type: "checkpoint-start" });
      const checkpoint = new codec.PngRows(width, height);
      const write = async (data) => {
        if (data.byteLength)
          await send({ type: "checkpoint-chunk", data }, [data.buffer]);
      };
      await write(checkpoint.drain());
      for (let y = 0; y < height; y += 128) {
        const rows = Math.min(128, height - y);
        checkpoint.push(
          repaired.rgb.subarray(y * width * 3, (y + rows) * width * 3),
          rows,
        );
        await write(checkpoint.drain());
      }
      await write(checkpoint.finish());
      await send({ type: "checkpoint-done", width, height });
    }
    if (operation === "repair") repairPixels = repaired.rgb;
    repaired = null;
  } else {
    canvas = originalCanvas;
    width = canvas.width;
    height = canvas.height;
    context = canvas.getContext("2d", {
      willReadFrequently: true,
      colorSpace: "srgb",
    });
  }

  const output = bounds(width, height, scale);
  const previewShape = previewSize(width, height);
  const ratio = previewShape.ratio;
  const preview = await createImageBitmap(canvas, {
    resizeWidth: previewShape.width,
    resizeHeight: previewShape.height,
  });
  await send(
    {
      type: "input",
      bitmap: preview,
      width,
      height,
      outputWidth: output.width,
      outputHeight: output.height,
      scale,
      detailCore: operation === "repair" ? 512 : CORE * scale,
    },
    [preview],
  );
  if (operation === "repair") {
    const started = performance.now();
    const encoder = new codec.PngRows(width, height);
    const emit = async (data) => {
      if (data.byteLength) await send({ type: "chunk", data }, [data.buffer]);
    };
    const detailCore = 512;
    const total =
      Math.ceil(width / detailCore) * Math.ceil(height / detailCore);
    let completed = 0,
      maxBandBytes = 0;
    await emit(encoder.drain());
    for (let y = 0; y < height; y += detailCore) {
      const rows = Math.min(detailCore, height - y);
      for (let x = 0; x < width; x += detailCore) {
        const columns = Math.min(detailCore, width - x),
          rgb = new Uint8Array(columns * rows * 3);
        for (let row = 0; row < rows; row++) {
          const offset = ((y + row) * width + x) * 3;
          rgb.set(
            repairPixels.subarray(offset, offset + columns * 3),
            row * columns * 3,
          );
        }
        const image = await createImageBitmap(
          new ImageData(rgbaFromRgb(rgb, columns, rows), columns, rows),
          {
            resizeWidth: Math.max(1, Math.ceil(columns * ratio)),
            resizeHeight: Math.max(1, Math.ceil(rows * ratio)),
          },
        );
        let png;
        if (detailTiles) {
          const tileEncoder = new codec.PngRows(columns, rows);
          tileEncoder.push(rgb, rows);
          png = tileEncoder.finish();
        }
        completed++;
        await send(
          {
            type: "tile",
            bitmap: image,
            x: x * ratio,
            y: y * ratio,
            width: columns * ratio,
            height: rows * ratio,
            completed,
            total,
            seconds: (performance.now() - started) / 1000,
            workers: 1,
            png,
            column: x / detailCore,
            row: y / detailCore,
          },
          png ? [image, png.buffer] : [image],
        );
      }
      // Preserve the original encoder strip/chunk boundaries independently of inspection tiles.
      for (let line = y; line < y + rows; line += CORE) {
        const stripRows = Math.min(CORE, y + rows - line);
        const band = repairPixels.subarray(
          line * width * 3,
          (line + stripRows) * width * 3,
        );
        maxBandBytes = Math.max(maxBandBytes, band.byteLength);
        encoder.push(band, stripRows);
        await emit(encoder.drain());
      }
    }
    await emit(encoder.finish());
    repairPixels = null;
    canvas.width = 1;
    canvas.height = 1;
    postMessage({
      type: "done",
      stats: {
        operation,
        backend: "wasm",
        model: "canon4",
        scale: 1,
        width,
        height,
        outputWidth: width,
        outputHeight: height,
        tiles: completed,
        seconds: (performance.now() - started) / 1000,
        processingSeconds: (performance.now() - jobStarted) / 1000,
        repair: repairReport,
        decode: decodeReport,
        workers: 1,
        workerReservationMiB: 0,
        peakInFlight: 1,
        gpu: null,
        wasmBytes: repairReport.wasmBytes,
        maxBandBytes,
        maxTileFloats: 0,
      },
    });
    return;
  }
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
        let png;
        if (detailTiles) {
          const tileEncoder = new codec.PngRows(
            tile.cw * scale,
            tile.ch * scale,
          );
          tileEncoder.push(rgb, tile.ch * scale);
          png = tileEncoder.finish();
        }
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
            png,
            column: x / CORE,
            row: y / CORE,
          },
          png ? [image, png.buffer] : [image],
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
      processingSeconds: (performance.now() - jobStarted) / 1000,
      maxBandBytes,
      maxTileFloats,
      repair: repairReport,
      decode: decodeReport,
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
