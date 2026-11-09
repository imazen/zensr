import { test } from "node:test";
import assert from "node:assert/strict";
import { bounds, tileAt, planar, paste, CORE, rgbaFromRgb } from "./core.mjs";
test("RGB source conversion respects strided input rows", () => {
  assert.deepEqual(
    [...rgbaFromRgb(new Uint8Array([1, 2, 3, 99, 4, 5, 6]), 1, 2, 4)],
    [1, 2, 3, 255, 4, 5, 6, 255],
  );
  assert.throws(() => rgbaFromRgb(new Uint8Array(4), 2, 1));
});
test("12MP admission and oversized rejection", () => {
  assert.deepEqual(bounds(4000, 3000, 4), { width: 16000, height: 12000 });
  assert.throws(() => bounds(8000, 8000, 4));
  assert.throws(() => bounds(200, 200, 3));
});
test("clamped halo covers tiny and odd image borders without invented padding", () => {
  assert.deepEqual(tileAt(0, 0, 5, 1, 21), {
    x: 0,
    y: 0,
    cw: 5,
    ch: 1,
    left: 0,
    top: 0,
    ew: 5,
    eh: 1,
  });
  const t = tileAt(128, 128, 131, 135, 16);
  assert.equal(t.left, 112);
  assert.equal(t.top, 112);
  assert.equal(t.ew, 19);
  assert.equal(t.eh, 23);
});
test("tile output places all rows and columns once", () => {
  const width = 131,
    height = 3,
    scale = 2;
  const band = new Uint8Array(width * height * scale * scale * 3);
  for (let x = 0; x < width; x += CORE) {
    const t = tileAt(x, 0, width, height, 16);
    paste(
      new Uint8Array(t.cw * t.ch * scale * scale * 3).fill(x ? 29 : 11),
      band,
      t,
      width,
      scale,
    );
  }
  for (let y = 0; y < height * scale; y++)
    for (let x = 0; x < width * scale; x++) {
      assert.equal(band[(y * width * scale + x) * 3], x < 256 ? 11 : 29);
    }
});
test("RGBA extraction preserves RGB order", () => {
  const out = planar(new Uint8Array([255, 0, 128, 255, 0, 255, 64, 255]), 2, 1);
  assert.deepEqual([...out.slice(0, 4)], [1, 0, 0, 1]);
  assert.equal(out[4], Math.fround(128 / 255));
});
