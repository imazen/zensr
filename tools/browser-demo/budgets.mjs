export const MIB = 1024 * 1024;
export const GPU_LIMIT = 256 * MIB;
export const WASM_LIMIT = 256 * MIB;
export const BLOB_LIMIT = 128 * MIB;
export function capWasmMemory(limit = WASM_LIMIT) {
  const Native = WebAssembly.Memory,
    memories = [];
  function Capped(descriptor) {
    const maximum = Math.min(
      descriptor.maximum ?? limit / 65536,
      limit / 65536,
    );
    if (descriptor.initial > maximum)
      throw new Error("Runtime exceeds the WASM memory budget.");
    const memory = new Native({ ...descriptor, maximum });
    memories.push(new WeakRef(memory));
    return memory;
  }
  Capped.prototype = Native.prototype;
  Object.setPrototypeOf(Capped, Native);
  WebAssembly.Memory = Capped;
  return () =>
    memories.reduce(
      (sum, ref) => sum + (ref.deref()?.buffer.byteLength || 0),
      0,
    );
}
export function capGpuBuffers(device, limit = GPU_LIMIT) {
  let live = 0,
    peak = 0;
  const create = device.createBuffer.bind(device);
  device.createBuffer = (descriptor) => {
    if (
      !Number.isSafeInteger(descriptor.size) ||
      descriptor.size < 0 ||
      live + descriptor.size > limit
    ) {
      throw new Error(
        "GPU buffer budget exceeded. Try WASM CPU or a smaller image.",
      );
    }
    const buffer = create(descriptor),
      destroy = buffer.destroy.bind(buffer);
    live += descriptor.size;
    peak = Math.max(peak, live);
    let released = false;
    buffer.destroy = () => {
      if (!released) {
        live -= descriptor.size;
        released = true;
      }
      destroy();
    };
    return buffer;
  };
  return () => ({ live, peak, limit });
}
