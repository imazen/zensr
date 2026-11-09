// End-to-end UI outputs from the same RGB8 fixtures as author PyTorch.
import { chromium } from "playwright";
import { readFile, writeFile, mkdir } from "node:fs/promises";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
const [root, backend = "wasm"] = process.argv.slice(2);
const cases = JSON.parse(await readFile(root + "/cases.json"));
const codec = await import("./codec/pkg/zensr_browser_codec.js");
await codec.default({
  module_or_path: await readFile(
    new URL("./codec/pkg/zensr_browser_codec_bg.wasm", import.meta.url),
  ),
});
for (const item of cases) {
  const bytes = await readFile(item.rgb);
  assert.equal(createHash("sha256").update(bytes).digest("hex"), item.sha256);
  const encoder = new codec.PngRows(item.width, item.height);
  encoder.push(bytes, item.height);
  await writeFile(root + "/" + item.id + ".png", encoder.finish());
}
const browser = await chromium.launch({
  executablePath: process.env.ZENSR_CHROME,
  headless: true,
  args: ["--no-sandbox", "--use-angle=swiftshader", "--enable-unsafe-webgpu"],
});
try {
  for (const item of cases)
    for (const model of ["spanf", "dscf"])
      for (const scale of [4, 2]) {
        const folder = `${root}/${item.id}-${model}-${backend}-x${scale}`;
        await mkdir(folder);
        const context = await browser.newContext({
          ignoreHTTPSErrors: true,
          acceptDownloads: true,
        });
        const page = await context.newPage();
        await page.goto(process.env.ZENSR_URL || "https://localhost:3331");
        await page.waitForFunction(
          () =>
            !document.getElementById("status").textContent.includes("Checking"),
        );
        await page
          .locator("#file")
          .setInputFiles(root + "/" + item.id + ".png");
        await page.selectOption("#model", model);
        await page.selectOption("#scale", String(scale));
        await page.selectOption("#backend", backend);
        if (backend === "wasm") await page.selectOption("#cpu-workers", "2");
        await page.click("#start");
        await page.waitForFunction(
          () => window.zensrState.done || window.zensrState.error,
          {},
          { timeout: 180000 },
        );
        const result = await page.evaluate(() => window.zensrState);
        await writeFile(
          folder + "/result.json",
          JSON.stringify(result, null, 2) + "\n",
        );
        assert.equal(result.error, null, JSON.stringify(result));
        assert.deepEqual(
          [result.stats.outputWidth, result.stats.outputHeight],
          [item.width * scale, item.height * scale],
        );
        const download = page.waitForEvent("download");
        await page.click("#download");
        await (await download).saveAs(folder + "/output.png");
        console.log(
          JSON.stringify({
            case: item.id,
            model,
            backend,
            scale,
            seconds: result.stats.processingSeconds,
          }),
        );
        await context.close();
      }
} finally {
  await browser.close();
}
