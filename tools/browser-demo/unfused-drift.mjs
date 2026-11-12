// Quantify, rather than assume, rounding drift from the authorized unfused fallback.
import assert from "node:assert/strict";
import { readFile, writeFile, mkdir } from "node:fs/promises";
import { createHash } from "node:crypto";
import { pathToFileURL } from "node:url";
import { resolve } from "node:path";
const [manifest, unfusedPackage, out] = process.argv.slice(2);
await mkdir(out, { recursive: true });
const strict = await import("./repair/pkg/zensr_browser_repair.js"),
  unfused = await import(
    pathToFileURL(resolve(unfusedPackage, "zensr_browser_repair.js"))
  );
await strict.default({
  module_or_path: await readFile(
    new URL("./repair/pkg/zensr_browser_repair_bg.wasm", import.meta.url),
  ),
});
await unfused.default({
  module_or_path: await readFile(
    resolve(unfusedPackage, "zensr_browser_repair_bg.wasm"),
  ),
});
const weights = await readFile(
    new URL("./dist/models/canon4.raw", import.meta.url),
  ),
  records = [];
for (const row of JSON.parse(await readFile(manifest, "utf8"))) {
  const bytes = await readFile(row.input);
  assert.equal(
    createHash("sha256").update(bytes).digest("hex"),
    row.input_sha256,
  );
  const a = strict.repair(bytes, weights),
    b = unfused.repair(bytes, weights);
  assert.deepEqual(
    [a.width, a.height, a.applied],
    [b.width, b.height, b.applied],
  );
  const applied = a.applied,
    rgb = a.take_pixels(),
    candidate = b.take_pixels();
  let different = 0,
    max = 0,
    sse = 0;
  for (let i = 0; i < rgb.length; i++) {
    const delta = Math.abs(rgb[i] - candidate[i]);
    if (delta) different++;
    max = Math.max(max, delta);
    sse += delta * delta;
  }
  records.push({ ...row, applied, values: rgb.length, different, max, sse });
  await writeFile(
    out + "/matrix.json",
    JSON.stringify(records, null, 2) + "\n",
  );
}
const summary = {
  cases: records.length,
  applied: records.filter((r) => r.applied).length,
  values: records.reduce((n, r) => n + r.values, 0),
  different: records.reduce((n, r) => n + r.different, 0),
  max: Math.max(...records.map((r) => r.max)),
  sse: records.reduce((n, r) => n + r.sse, 0),
};
await writeFile(out + "/summary.json", JSON.stringify(summary, null, 2) + "\n");
console.log(summary);
