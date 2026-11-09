export const MAX_PIXELS = 12_600_000;
export const CORE = 128;
export const HALO = { spanf: 16, dscf: 21 };
export function bounds(width, height, scale) {
  if (
    !Number.isInteger(width) ||
    !Number.isInteger(height) ||
    width < 1 ||
    height < 1 ||
    width > 8192 ||
    height > 8192 ||
    width * height > MAX_PIXELS ||
    ![2, 4].includes(scale)
  ) {
    throw new Error("Choose an image up to 12.6 MP and 8192 pixels per side.");
  }
  return { width: width * scale, height: height * scale };
}
export function tileAt(x, y, width, height, halo) {
  const x1 = Math.min(x + CORE, width),
    y1 = Math.min(y + CORE, height);
  const left = Math.max(0, x - halo),
    top = Math.max(0, y - halo);
  const right = Math.min(width, x1 + halo),
    bottom = Math.min(height, y1 + halo);
  return {
    x,
    y,
    cw: x1 - x,
    ch: y1 - y,
    left,
    top,
    ew: right - left,
    eh: bottom - top,
  };
}
export function planar(rgba, width, height) {
  if (rgba.length !== width * height * 4)
    throw new Error("Decoded tile shape differs.");
  const n = width * height,
    out = new Float32Array(3 * n);
  for (let i = 0; i < n; i++)
    for (let c = 0; c < 3; c++) out[c * n + i] = rgba[4 * i + c] / 255;
  return out;
}
export function paste(rgb, band, tile, width, scale) {
  const row = tile.cw * scale * 3;
  if (rgb.length !== row * tile.ch * scale)
    throw new Error("Output tile shape differs.");
  for (let y = 0; y < tile.ch * scale; y++) {
    band.set(
      rgb.subarray(y * row, (y + 1) * row),
      (y * width * scale + tile.x * scale) * 3,
    );
  }
}
