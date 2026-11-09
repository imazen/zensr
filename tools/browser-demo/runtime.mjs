import { capGpuBuffers } from "./budgets.mjs";
export async function modelBytes(model, codec) {
  const manifest = await (await fetch("./models.json")).json();
  const entry = manifest[model];
  if (!entry) throw new Error("Unknown model.");
  const bytes = new Uint8Array(await (await fetch(entry.url)).arrayBuffer());
  if (codec.sha256(bytes) !== entry.sha256)
    throw new Error("Model bytes failed verification.");
  return bytes;
}
export async function loadSession(bytes, backend) {
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
