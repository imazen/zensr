import { test } from "node:test";
import assert from "node:assert/strict";
import { capGpuBuffers, capWasmMemory } from "./budgets.mjs";
test("WASM constructor and growth enforce the memory maximum", () => {
  const Native = WebAssembly.Memory;
  try {
    const stats = capWasmMemory(3 * 65536);
    assert.throws(() => new WebAssembly.Memory({ initial: 4 }));
    const memory = new WebAssembly.Memory({ initial: 1, maximum: 100 });
    memory.grow(2);
    assert.equal(stats(), 3 * 65536);
    assert.throws(() => memory.grow(1), RangeError);
    assert.ok(memory instanceof Native);
  } finally {
    WebAssembly.Memory = Native;
  }
});
test("GPU accounting refuses over-budget allocations before allocating", () => {
  let calls = 0;
  const device = {
    createBuffer: () => {
      calls++;
      return { destroy() {} };
    },
  };
  const stats = capGpuBuffers(device, 100);
  const a = device.createBuffer({ size: 60 });
  assert.throws(() => device.createBuffer({ size: 41 }));
  assert.equal(calls, 1);
  const b = device.createBuffer({ size: 40 });
  assert.equal(stats().peak, 100);
  a.destroy();
  a.destroy();
  assert.equal(stats().live, 40);
  b.destroy();
  assert.equal(stats().live, 0);
});
