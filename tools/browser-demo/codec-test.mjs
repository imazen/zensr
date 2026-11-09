import assert from "node:assert/strict";
import { readFile, open, mkdir } from "node:fs/promises";
const out = process.argv[2];
if (!out) throw new Error("Pass output directory.");
await mkdir(out, { recursive: true });
const codec = await import("./codec/pkg/zensr_browser_codec.js");
await codec.default({
  module_or_path: await readFile(
    new URL("./codec/pkg/zensr_browser_codec_bg.wasm", import.meta.url),
  ),
});
const floats = new Float32Array(3 * 4 * 4);
for (let c = 0; c < 3; c++)
  for (let y = 0; y < 4; y++)
    for (let x = 0; x < 4; x++)
      floats[c * 16 + y * 4 + x] = (c * 50 + y * 4 + x) / 255;
assert.deepEqual(
  [...codec.pack_tile(floats, 4, 4, 1, 1, 2, 2, 4)],
  [5, 55, 105, 6, 56, 106, 9, 59, 109, 10, 60, 110],
);
assert.deepEqual(
  [
    ...codec.pack_tile(
      floats.map((v) => (v * 255) / 256),
      4,
      4,
      0,
      0,
      4,
      4,
      2,
    ),
  ],
  [2, 52, 102, 4, 54, 104, 10, 60, 110, 12, 62, 112],
);
assert.throws(() =>
  codec.pack_tile(new Float32Array([NaN]), 1, 1, 0, 0, 1, 1, 4),
);
const encoder = new codec.PngRows(4000, 3000),
  file = await open(out + "/input-12mp.png", "w");
try {
  await file.write(encoder.drain());
  for (let y = 0; y < 3000; y += 64) {
    const rows = Math.min(64, 3000 - y),
      band = new Uint8Array(4000 * rows * 3);
    for (let j = 0; j < rows; j++)
      for (let x = 0; x < 4000; x++) {
        const i = (j * 4000 + x) * 3;
        band[i] = (x + (y + j) * 3) % 256;
        band[i + 1] = (x * 3 + (y + j)) % 256;
        band[i + 2] = (x ^ (y + j)) % 256;
      }
    encoder.push(band, rows);
    await file.write(encoder.drain());
  }
  await file.write(encoder.finish());
} finally {
  await file.close();
}
assert.deepEqual(
  [...codec.dimensions(await readFile(out + "/input-12mp.png"))],
  [4000, 3000],
);
assert.throws(() => new codec.PngRows(0, 10));
const incomplete = new codec.PngRows(2, 2);
assert.throws(() => incomplete.finish());
console.log(
  "Float crop/resize, invalid inputs, strip encoder and 12MP fixture passed.",
);
