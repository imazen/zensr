// A separate, window-owned worker releases the repair heap before SR starts.
import { repairRuntime } from "./repair-runtime.mjs";
self.onmessage = async ({ data: { port, jpeg, weights } }) => {
  try {
    const { codec, math, fallbackReason } = await repairRuntime();
    const result = codec.repair(jpeg, weights);
    const report = {
      math,
      fallbackReason,
      width: result.width,
      height: result.height,
      applied: result.applied,
      reason: result.reason,
      wasmBytes: codec.memory_bytes(),
    };
    const rgb = result.take_pixels();
    port.postMessage({ ...report, rgb }, [rgb.buffer]);
  } catch (error) {
    port.postMessage({
      error: `JPEG repair failed: ${error.message || error}. Repair has a 1 GiB WASM cap.`,
    });
  }
};
