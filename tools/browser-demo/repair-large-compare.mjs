import assert from "node:assert/strict";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { chromium } from "playwright";
const [input, goldenJson, out] = process.argv.slice(2);
await mkdir(out, { recursive: true });
const golden = JSON.parse(await readFile(goldenJson, "utf8"));
const browser = await chromium.launch({
  executablePath: process.env.ZENSR_CHROME,
  headless: true,
  args: ["--no-sandbox", "--use-angle=swiftshader"],
});
try {
  const page = await browser.newPage({ ignoreHTTPSErrors: true });
  await page.goto("https://localhost:3331");
  await page.locator("#file").setInputFiles(input);
  const result = await page.evaluate(async () => {
    const module = await import("./repair-relaxed/zensr_browser_repair.js");
    await module.default();
    const jpeg = new Uint8Array(
        await document.getElementById("file").files[0].arrayBuffer(),
      ),
      weights = new Uint8Array(
        await (await fetch("./models/canon4.raw")).arrayBuffer(),
      );
    const start = performance.now(),
      value = module.repair(jpeg, weights),
      seconds = (performance.now() - start) / 1000;
    const result = {
        width: value.width,
        height: value.height,
        applied: value.applied,
        seconds,
        wasmBytes: module.memory_bytes(),
      },
      rgb = value.take_pixels();
    result.values = rgb.length;
    result.sha256 = [
      ...new Uint8Array(await crypto.subtle.digest("SHA-256", rgb)),
    ]
      .map((x) => x.toString(16).padStart(2, "0"))
      .join("");
    return result;
  });
  await writeFile(
    out + "/comparison.json",
    JSON.stringify({ golden, relaxed: result }, null, 2) + "\n",
  );
  console.log(result);
  assert.equal(
    result.sha256,
    golden.sha256,
    "All RGB8 values must match the retained strict output",
  );
  assert.equal(result.applied, true);
  assert.equal(result.values, 36_000_000);
} finally {
  await browser.close();
}
