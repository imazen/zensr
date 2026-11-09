import { chromium } from "playwright";
import { writeFile } from "node:fs/promises";
const browser = await chromium.launch({
  executablePath: process.env.ZENSR_CHROME,
  headless: true,
  args: ["--no-sandbox", "--use-angle=swiftshader", "--enable-unsafe-webgpu"],
});
try {
  const page = await browser.newPage({ ignoreHTTPSErrors: true });
  await page.goto("https://localhost:3331");
  const result = await page.evaluate(async () => {
    const ort = await import("./vendor/ort.wasm.bundle.min.mjs");
    ort.env.wasm.numThreads = 1;
    const core = await import("./core.mjs");
    const codec = await import("./codec/zensr_browser_codec.js");
    await codec.default();
    const records = [];
    for (const model of ["spanf", "dscf"]) {
      const session = await ort.InferenceSession.create(
        "./models/" + model + ".onnx",
        {
          executionProviders: ["wasm"],
          enableCpuMemArena: false,
          enableMemPattern: false,
        },
      );
      for (const [w, h] of [
        [1, 3],
        [259, 263],
      ]) {
        const input = new Float32Array(3 * w * h);
        let seed = 17;
        for (let i = 0; i < input.length; i++) {
          seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
          input[i] = (seed >>> 24) / 255;
        }
        const run = async (data, width, height) => {
          const t = new ort.Tensor("float32", data, [1, 3, height, width]);
          const r = await session.run({ [session.inputNames[0]]: t });
          const v = r[session.outputNames[0]].data.slice();
          Object.values(r).forEach((t) => t.dispose());
          t.dispose();
          return v;
        };
        const full = await run(input, w, h);
        let max = 0,
          different = 0,
          count = 0;
        for (let y = 0; y < h; y += core.CORE)
          for (let x = 0; x < w; x += core.CORE) {
            const t = core.tileAt(x, y, w, h, core.HALO[model]),
              local = new Float32Array(3 * t.ew * t.eh);
            for (let c = 0; c < 3; c++)
              for (let j = 0; j < t.eh; j++)
                local.set(
                  input.subarray(
                    c * w * h + (j + t.top) * w + t.left,
                    c * w * h + (j + t.top) * w + t.left + t.ew,
                  ),
                  c * t.ew * t.eh + j * t.ew,
                );
            const got = await run(local, t.ew, t.eh);
            for (let c = 0; c < 3; c++)
              for (let j = 0; j < t.ch * 4; j++)
                for (let i = 0; i < t.cw * 4; i++) {
                  const a =
                      got[
                        c * t.ew * t.eh * 16 +
                          (j + (y - t.top) * 4) * t.ew * 4 +
                          i +
                          (x - t.left) * 4
                      ],
                    b = full[c * w * h * 16 + (y * 4 + j) * w * 4 + x * 4 + i];
                  max = Math.max(max, Math.abs(a - b));
                  if (a !== b) different++;
                  count++;
                }
          }
        records.push({
          model,
          w,
          h,
          maxFloatDifference: max,
          different,
          count,
        });
        if (max !== 0) throw new Error(JSON.stringify(records));
      }
      await session.release();
    }
    return records;
  });
  await writeFile(process.argv[2], JSON.stringify(result, null, 2) + "\n");
  console.log(result);
} finally {
  await browser.close();
}
