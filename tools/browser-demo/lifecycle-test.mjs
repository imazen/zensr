import assert from "node:assert/strict";
import { chromium } from "playwright";
import { writeFile } from "node:fs/promises";
const [small, large, largeJpeg, out] = process.argv.slice(2);
const browser = await chromium.launch({
  executablePath: process.env.ZENSR_CHROME,
  headless: true,
  args: ["--no-sandbox", "--use-angle=swiftshader"],
});
try {
  const page = await browser.newPage({ ignoreHTTPSErrors: true });
  await page.goto("https://localhost:3331");
  await page.waitForFunction(
    () => !document.getElementById("status").textContent.includes("Checking"),
  );
  await page.selectOption("#backend", "wasm");
  await page.selectOption("#cpu-workers", "4");
  await page.locator("#file").setInputFiles(small);
  await page.click("#start");
  await page.waitForFunction(
    () => window.zensrState.done || window.zensrState.error,
  );
  const first = await page.evaluate(() => window.zensrState);
  assert.equal(first.error, null);
  assert.equal(first.done, true);
  const files = await page.evaluate(async () => {
    const root = await navigator.storage.getDirectory(),
      r = [];
    for await (const [n] of root.entries()) r.push(n);
    return r;
  });
  // Cancel while the separately owned native repair worker is active.
  await page.locator("#file").setInputFiles(largeJpeg);
  await page.locator("#repair").check();
  await page.click("#start");
  await page.waitForFunction(() =>
    document.getElementById("status").textContent.includes("Repairing"),
  );
  const repairDeadline = Date.now() + 5000;
  while (page.workers().length < 2) {
    if (Date.now() > repairDeadline)
      throw new Error("Repair worker was not created");
    await page.waitForTimeout(10);
  }
  await page.click("#cancel");
  await page.waitForFunction(() => !document.getElementById("start").disabled);
  const closedDeadline = Date.now() + 5000;
  while (page.workers().length) {
    if (Date.now() > closedDeadline)
      throw new Error("Cancelled repair worker remained alive");
    await page.waitForTimeout(25);
  }
  await page.locator("#file").setInputFiles(large);
  await page.click("#start");
  await page.waitForFunction(
    () => Number(document.querySelector("#progress").value) > 0,
  );
  assert.ok(page.workers().length >= 2);
  await page.click("#cancel");
  await page.waitForFunction(() => !document.getElementById("start").disabled);
  assert.match(await page.textContent("#status"), /Cancelled/);
  const deadline = Date.now() + 5000;
  while (page.workers().length) {
    if (Date.now() > deadline)
      throw new Error("Cancelled tile workers remained alive");
    await page.waitForTimeout(25);
  }
  const preserved = await page.evaluate(async (name) => {
    const root = await navigator.storage.getDirectory();
    return (await (await root.getFileHandle(name)).getFile()).size;
  }, first.stats.outputName);
  assert.equal(preserved, first.stats.encodedBytes);
  await page.locator("#file").setInputFiles(small);
  if (await page.locator("#repair").isEnabled())
    await page.locator("#repair").uncheck();
  await page.click("#start");
  await page.waitForFunction(
    () => window.zensrState.done || window.zensrState.error,
  );
  assert.equal(await page.evaluate(() => window.zensrState.error), null);
  await page.reload();
  await page.waitForFunction(
    () => document.querySelectorAll("#saved a").length === 2,
  );
  assert.equal(await page.locator("#saved a").count(), 2);
  // Malformed bytes fail explicitly and do not disturb saved outputs.
  await page.locator("#file").setInputFiles({
    name: "broken.png",
    mimeType: "image/png",
    buffer: Buffer.from("broken"),
  });
  await page.click("#start");
  await page.waitForFunction(() => window.zensrState.error);
  assert.equal(await page.locator("#saved a").count(), 2);
  const result = {
    cancelled: true,
    repairCancelled: true,
    allWorkersTerminated: true,
    restarted: true,
    savedFilesAfterReload: 2,
    originalBytesPreserved: preserved,
    malformedRejected: true,
  };
  await writeFile(out, JSON.stringify(result, null, 2) + "\n");
  console.log(result);
} finally {
  await browser.close();
}
