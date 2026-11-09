import { capWasmMemory } from "./budgets.mjs";
import { WORKER_WASM_MIB } from "./pool-plan.mjs";
import { loadSession } from "./runtime.mjs";
const memoryBytes = capWasmMemory(WORKER_WASM_MIB * 1048576);
self.onmessage = async ({ data: { port, bytes } }) => {
  try {
    const { ort, session } = await loadSession(bytes, "wasm");
    port.onmessage = async ({ data: { data, width, height } }) => {
      try {
        const input = new ort.Tensor("float32", data, [1, 3, height, width]);
        const result = await session.run({ [session.inputNames[0]]: input });
        const output = result[session.outputNames[0]];
        if (output.dims.join(",") !== [1, 3, height * 4, width * 4].join(","))
          throw new Error("Model tile shape differs.");
        // Transfer an owned output; disposing the runtime tensor must not detach its heap.
        const floats = output.data.slice();
        Object.values(result).forEach((t) => t.dispose());
        input.dispose();
        port.postMessage({ floats, wasmBytes: memoryBytes() }, [floats.buffer]);
      } catch (error) {
        port.postMessage({ error: String(error.message || error) });
      }
    };
    port.postMessage({ ready: true, wasmBytes: memoryBytes() });
  } catch (error) {
    port.postMessage({ error: String(error.message || error) });
  }
};
