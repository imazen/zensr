import assert from "node:assert/strict";
import { mkdir, writeFile } from "node:fs/promises";
import { chromium } from "playwright";
const [input, out] = process.argv.slice(2);
await mkdir(out, { recursive: true });
const browser = await chromium.launch({
  executablePath: process.env.ZENSR_CHROME,
  headless: true,
  args: ["--no-sandbox", "--use-angle=swiftshader"],
});
try {
  const page = await browser.newPage({
    ignoreHTTPSErrors: true,
    acceptDownloads: true,
  });
  await page.addInitScript(() => {
    const create = FileSystemFileHandle.prototype.createWritable;
    FileSystemFileHandle.prototype.createWritable = function (...args) {
      if (
        document.getElementById("repair")?.checked &&
        !window.zensrState?.repairReport
      )
        throw new Error("Output file was opened before repair completed");
      return create.apply(this, args);
    };
  });
  await page.goto("https://localhost:3331");
  await page.waitForFunction(
    () => !document.getElementById("status").textContent.includes("Checking"),
  );
  await page.locator("#file").setInputFiles(input);
  await page.locator("#repair").check();
  await page.selectOption("#backend", "wasm");
  await page.selectOption("#cpu-workers", "1");
  await page.click("#start");
  await page.waitForFunction(
    () => window.zensrState.done || window.zensrState.error,
  );
  const result = await page.evaluate(() => window.zensrState);
  assert.equal(result.error, null);
  assert.equal(result.done, true);
  assert.ok(result.checkpoint.bytes > 0);
  assert.ok(result.repairReport.applied);
  const download = page.waitForEvent("download");
  await page.click("#repair-download");
  await (await download).saveAs(out + "/jpeg-stage.png");
  await page.reload();
  await page.waitForFunction(
    () => document.querySelectorAll("#saved a").length === 2,
  );
  await writeFile(
    out + "/writer.json",
    JSON.stringify(
      { lateOutputCreation: true, checkpointPersists: true, ...result },
      null,
      2,
    ) + "\n",
  );
  console.log({
    lateOutputCreation: true,
    checkpointBytes: result.checkpoint.bytes,
  });
} finally {
  await browser.close();
}
