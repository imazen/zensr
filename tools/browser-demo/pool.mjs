// Ports connect to window-owned workers, so cancel can terminate every worker directly.
export class TilePool {
  constructor(ports) {
    this.inFlight = 0;
    this.peakInFlight = 0;
    this.slots = ports.map((port) => {
      let resolveReady, rejectReady;
      const slot = {
        port,
        busy: false,
        bytes: 0,
        ready: new Promise((resolve, reject) => {
          resolveReady = resolve;
          rejectReady = reject;
        }),
      };
      port.onmessage = ({ data }) => {
        if (data.error) {
          const e = new Error(data.error);
          rejectReady(e);
          slot.reject?.(e);
          return;
        }
        slot.bytes = data.wasmBytes;
        if (data.ready) {
          resolveReady();
          return;
        }
        slot.busy = false;
        this.inFlight--;
        slot.resolve(data.floats);
        slot.resolve = null;
        slot.reject = null;
      };
      port.onmessageerror = () => {
        const e = new Error("CPU tile message could not be decoded.");
        rejectReady(e);
        slot.reject?.(e);
      };
      return slot;
    });
  }
  async ready() {
    await Promise.all(this.slots.map((slot) => slot.ready));
  }
  run(index, data, width, height) {
    const slot = this.slots[index % this.slots.length];
    if (slot.busy)
      throw new Error("CPU tile queue exceeded its bounded window.");
    slot.busy = true;
    this.inFlight++;
    this.peakInFlight = Math.max(this.peakInFlight, this.inFlight);
    return new Promise((resolve, reject) => {
      slot.resolve = resolve;
      slot.reject = reject;
      slot.port.postMessage({ data, width, height }, [data.buffer]);
    });
  }
  memoryBytes() {
    return this.slots.reduce((sum, s) => sum + s.bytes, 0);
  }
  close() {
    this.slots.forEach((s) => s.port.close());
  }
}
