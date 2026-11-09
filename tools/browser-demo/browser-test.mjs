// Explicit browser, input, backend and output arguments keep these tests reproducible.
import { chromium } from "playwright";
import { writeFile, mkdir } from "node:fs/promises";
import assert from "node:assert/strict";
const [file, model, scale, backend, out] = process.argv.slice(2);
if (!file || !out || !process.env.ZENSR_CHROME)
  throw new Error(
    "Use browser-test.mjs input model scale backend output-directory; set ZENSR_CHROME.",
  );
await mkdir(out, { recursive: true });
const browser = await chromium.launch({
  executablePath: process.env.ZENSR_CHROME,
  headless: true,
  args: [
    "--no-sandbox",
    "--enable-unsafe-webgpu",
    ...(process.env.ZENSR_GPU_FLAGS || "--use-angle=swiftshader").split(" "),
  ],
});
try {
  const page = await browser.newPage({
    ignoreHTTPSErrors: true,
    acceptDownloads: true,
    viewport: { width: 1400, height: 1000 },
  });
  page.on("pageerror", (e) => console.error("PAGE", e.message));
  page.on("console", (m) => console.log(m.type(), m.text()));
  const requests = [];
  page.on("request", (r) =>
    requests.push({ method: r.method(), url: r.url() }),
  );
  await page.goto(process.env.ZENSR_URL || "https://localhost:3331");
  await page.waitForFunction(
    () => !document.getElementById("status").textContent.includes("Checking"),
  );
  await page.locator("#file").setInputFiles(file);
  await page.selectOption("#model", model);
  await page.selectOption("#scale", scale);
  await page.selectOption("#backend", backend);
  if (process.env.ZENSR_CPU_WORKERS)
    await page.selectOption("#cpu-workers", process.env.ZENSR_CPU_WORKERS);
  if (await page.locator("#repair").isEnabled())
    await page.locator("#repair").setChecked(process.env.ZENSR_REPAIR === "1");
  await page.exposeFunction("recordProgress", async (progress) =>
    writeFile(
      out + "/progress.json",
      JSON.stringify(
        { pid: process.pid, ...progress, lastCheck: new Date().toISOString() },
        null,
        2,
      ) + "\n",
    ),
  );
  await page.evaluate(() => {
    new MutationObserver(() => {
      const text = document.getElementById("status").textContent;
      const match = /^(\d+) \/ (\d+) tiles/.exec(text);
      if (!match || Number(match[1]) % 32 === 0)
        window.recordProgress({
          status: text,
          repair: window.zensrState.repairReport,
          checkpoint: window.zensrState.checkpoint,
        });
    }).observe(document.getElementById("status"), { childList: true });
  });
  await page.click(
    process.env.ZENSR_OPERATION === "repair" ? "#repair-only" : "#start",
  );
  await page.waitForFunction(
    () => window.zensrState.done || window.zensrState.error,
    {},
    { timeout: 1800000 },
  );
  const result = await page.evaluate(() => window.zensrState);
  await page.screenshot({ path: out + "/browser.png", fullPage: true });
  await writeFile(
    out + "/result.json",
    JSON.stringify({ ...result, requests }, null, 2) + "\n",
  );
  if (result.checkpoint) {
    const download = page.waitForEvent("download");
    await page.click("#repair-download");
    await (await download).saveAs(out + "/jpeg-stage.png");
  }
  if (result.error) throw new Error(result.error);
  assert.ok(
    requests.every((r) => r.method === "GET"),
    "Images must not be uploaded",
  );
  const downloaded = page.waitForEvent("download");
  await page.click("#download");
  await (await downloaded).saveAs(out + "/output.png");
  console.log(JSON.stringify(result));
  if (process.env.ZENSR_INSPECT_NATIVE === "1") {
    await page.click("#zoom-native");
    const inspected = await page.evaluate(() => ({
      core: window.zensrViewer.core,
      visible: window.zensrViewer.visible().length,
      ...window.zensrViewer.stats,
    }));
    await writeFile(
      out + "/native-inspection.json",
      JSON.stringify(inspected, null, 2) + "\n",
    );
    assert.ok(
      inspected.visible > 0,
      "At 100%, repair output must admit native tiles rather than the reduced preview",
    );
    await page.waitForFunction(
      () =>
        window.zensrViewer
          .visible()
          .every((t) => window.zensrViewer.cache.has(t.key)) &&
        window.zensrViewer.stats.loading === 0,
    );
    await page.screenshot({ path: out + "/native.png", fullPage: true });
    const witnesses = await page.evaluate(async () => {
      const viewer = window.zensrViewer,
        result = [];
      for (const tile of viewer.visible()) {
        const file = await (
          await viewer.directory.getFileHandle(tile.key)
        ).getFile();
        result.push({
          name: tile.key,
          bytes: [...new Uint8Array(await file.arrayBuffer())],
        });
      }
      return result;
    });
    for (const tile of witnesses)
      await writeFile(out + "/" + tile.name, Buffer.from(tile.bytes));
  }
} finally {
  await browser.close();
}
