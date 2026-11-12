// Both JPEG operations use the same native codec/runtime module and capability selection.
import { repairRuntime } from "./repair-runtime.mjs";
self.onmessage = async ({ data: { port, jpeg, weights, operation } }) => {
  try {
    const { codec, math, fallbackReason } = await repairRuntime();
    codec.dimensions(jpeg);
    const result =
      operation === "decode"
        ? codec.decode(jpeg)
        : operation === "repair"
          ? codec.repair(jpeg, weights)
          : (() => {
              throw new Error("Unknown JPEG operation");
            })();
    const report = {
      width: result.width,
      height: result.height,
      wasmBytes: codec.memory_bytes(),
    };
    if (operation === "repair")
      Object.assign(report, {
        applied: result.applied,
        reason: result.reason,
        math,
        fallbackReason,
      });
    const rgb = result.take_pixels();
    if (operation === "decode")
      port.postMessage({ ...report, rgb, jpeg }, [rgb.buffer, jpeg.buffer]);
    else port.postMessage({ ...report, rgb }, [rgb.buffer]);
  } catch (error) {
    port.postMessage({
      error: `JPEG ${operation} failed: ${error.message || error}`,
    });
  }
};
