import assert from "node:assert/strict";
import { readFile, writeFile, mkdir } from "node:fs/promises";
import { createHash } from "node:crypto";
const [jpegPath, weightsPath, oraclePath, profiledPath, out] =
  process.argv.slice(2);
await mkdir(out, { recursive: true });
const module = await import("./repair/pkg/zensr_browser_repair.js");
await module.default({
  module_or_path: await readFile(
    new URL("./repair/pkg/zensr_browser_repair_bg.wasm", import.meta.url),
  ),
});
const jpeg = await readFile(jpegPath),
  weights = await readFile(weightsPath),
  oracle = await readFile(oraclePath);
const base = module.repair(jpeg, weights),
  width = base.width,
  height = base.height,
  applied = base.applied;
const pixels = base.take_pixels();
assert.deepEqual(
  Buffer.from(pixels),
  oracle,
  "WASM/native public API RGB8 parity",
);
const awaitProfiled = await readFile(profiledPath);
assert.throws(
  () => module.repair(awaitProfiled, weights),
  /embedded ICC profiles/,
);
const results = [];
for (let orientation = 1; orientation <= 8; orientation++) {
  // An independent, minimal EXIF test fixture. Runtime parsing uses the shared codec.
  const payload = Buffer.from([
    69,
    120,
    105,
    102,
    0,
    0,
    73,
    73,
    42,
    0,
    8,
    0,
    0,
    0,
    1,
    0,
    18,
    1,
    3,
    0,
    1,
    0,
    0,
    0,
    orientation,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
  ]);
  const bytes = Buffer.concat([
    jpeg.subarray(0, 2),
    Buffer.from([255, 225, 0, payload.length + 2]),
    payload,
    jpeg.subarray(2),
  ]);
  const result = module.repair(bytes, weights),
    w = result.width,
    h = result.height;
  assert.equal(result.applied, applied);
  assert.deepEqual(
    [w, h],
    orientation >= 5 ? [height, width] : [width, height],
  );
  const actual = result.take_pixels(),
    expected = new Uint8Array(pixels.length);
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++) {
      const maps = [
        [x, y],
        [width - 1 - x, y],
        [width - 1 - x, height - 1 - y],
        [x, height - 1 - y],
        [y, x],
        [height - 1 - y, x],
        [height - 1 - y, width - 1 - x],
        [y, width - 1 - x],
      ];
      const [dx, dy] = maps[orientation - 1];
      expected.set(
        pixels.subarray((y * width + x) * 3, (y * width + x + 1) * 3),
        (dy * w + dx) * 3,
      );
    }
  assert.deepEqual(actual, expected, `EXIF orientation ${orientation}`);
  await writeFile(out + `/orientation-${orientation}.jpg`, bytes);
  await writeFile(out + `/orientation-${orientation}.rgb8`, actual);
  results.push({
    orientation,
    width: w,
    height: h,
    applied,
    sha256: createHash("sha256").update(actual).digest("hex"),
  });
}
assert.throws(() => module.repair(jpeg, new Uint8Array(2)));
await writeFile(
  out + "/repair.json",
  JSON.stringify(
    { nativeRgb8Exact: true, profiledRejected: true, orientations: results },
    null,
    2,
  ) + "\n",
);
console.log(results);
