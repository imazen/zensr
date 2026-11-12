import { test } from "node:test";
import assert from "node:assert/strict";
import { repairRuntime } from "./repair-runtime.mjs";
test("repair always tries relaxed first and does not initialize a second heap on success", async () => {
  const attempts = [],
    codec = {};
  const result = await repairRuntime(async (mode) => {
    attempts.push(mode);
    return codec;
  });
  assert.deepEqual(attempts, ["relaxed"]);
  assert.equal(result.codec, codec);
  assert.equal(result.math, "relaxed");
  assert.equal(result.fallbackReason, null);
});
test("unsupported relaxed arithmetic falls back to unfused, with its reason retained", async () => {
  const attempts = [],
    codec = {};
  const result = await repairRuntime(async (mode) => {
    attempts.push(mode);
    if (mode === "relaxed")
      throw new WebAssembly.CompileError("unsupported relaxed opcode");
    return codec;
  });
  assert.deepEqual(attempts, ["relaxed", "unfused"]);
  assert.equal(result.codec, codec);
  assert.equal(result.math, "unfused");
  assert.match(result.fallbackReason, /unsupported relaxed opcode/);
});
test("both initialization failures propagate without selecting slow portable fused arithmetic", async () => {
  const attempts = [];
  await assert.rejects(
    repairRuntime(async (mode) => {
      attempts.push(mode);
      throw new Error(`${mode} failed`);
    }),
    /relaxed failed.*unfused failed/,
  );
  assert.deepEqual(attempts, ["relaxed", "unfused"]);
});
