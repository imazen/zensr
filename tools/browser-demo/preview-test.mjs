// Small SR outputs must retain native overview samples within the 4 MP budget.
import assert from "node:assert/strict";
import { chromium } from "playwright";
import { writeFile, readFile } from "node:fs/promises";
const [input, out, expectedOutput] = process.argv.slice(2);
const browser = await chromium.launch({
  executablePath: process.env.ZENSR_CHROME,
  headless: true,
  args: ["--no-sandbox"],
});
try {
  const page = await browser.newPage({ ignoreHTTPSErrors: true });
  await page.goto(process.env.ZENSR_URL || "https://localhost:3331");
  await page.waitForFunction(
    () => !document.getElementById("status").textContent.includes("Checking"),
  );
  await page.locator("#file").setInputFiles(input);
  await page.selectOption("#model", "spanf");
  await page.selectOption("#scale", "4");
  await page.selectOption("#backend", "wasm");
  await page.selectOption("#cpu-workers", "2");
  await page.click("#start");
  await page.waitForFunction(
    () => window.zensrState.done || window.zensrState.error,
  );
  const result = await page.evaluate(() => ({
    state: window.zensrState,
    width: document.getElementById("after").width,
    height: document.getElementById("after").height,
  }));
  await writeFile(out, JSON.stringify(result, null, 2) + "\n");
  assert.equal(result.state.error, null);
  const { outputWidth: w, outputHeight: h } = result.state.stats;
  assert.ok(
    w * h <= 4_000_000 && w <= 4096 && h <= 4096,
    "Witness must fit overview budget",
  );
  assert.deepEqual(
    [result.width, result.height],
    [w, h],
    "Small SR overview must retain full output dimensions",
  );
  if (expectedOutput) {
    const download = page.waitForEvent("download");
    await page.click("#download");
    const path = await (await download).path();
    assert.deepEqual(
      await readFile(path),
      await readFile(expectedOutput),
      "Overview fix must preserve downloaded PNG bytes",
    );
  }
  console.log({ preview: [result.width, result.height], nativeOverview: true });
} finally {
  await browser.close();
}
