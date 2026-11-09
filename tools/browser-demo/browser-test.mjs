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
  if(process.env.ZENSR_CPU_WORKERS)await page.selectOption("#cpu-workers",process.env.ZENSR_CPU_WORKERS);
  await page.click("#start");
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
  if (result.error) throw new Error(result.error);
  assert.ok(
    requests.every((r) => r.method === "GET"),
    "Images must not be uploaded",
  );
  const downloaded = page.waitForEvent("download");
  await page.click("#download");
  await (await downloaded).saveAs(out + "/output.png");
  console.log(JSON.stringify(result));
} finally {
  await browser.close();
}
