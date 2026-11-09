import assert from "node:assert/strict";
import { mkdir, writeFile } from "node:fs/promises";
import { chromium } from "playwright";
const out = process.argv[2];
await mkdir(out, { recursive: true });
const browser = await chromium.launch({
  executablePath: process.env.ZENSR_CHROME,
  headless: true,
  args: ["--no-sandbox", "--use-angle=swiftshader"],
});
try {
  const page = await browser.newPage({
      ignoreHTTPSErrors: true,
      viewport: { width: 1450, height: 1000 },
    }),
    errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("https://localhost:3331/gallery/witnesses.html");
  await page.waitForFunction(() =>
    document
      .querySelector("section figure:nth-child(3) a")
      .search.includes("candidate="),
  );
  const href = await page
    .locator("section")
    .first()
    .locator("figure a")
    .nth(2)
    .getAttribute("href");
  await page.goto("https://localhost:3331" + href);
  await page.waitForFunction(() =>
    document.getElementById("status").textContent.includes("Gallery ready"),
  );
  await page.click("#zoom-native");
  assert.equal(await page.evaluate(() => window.zensrViewer.stats.zoom), 1);
  assert.equal(
    await page.evaluate(() => window.zensrViewer.stats.detail),
    true,
  );
  assert.ok(await page.evaluate(() => !!window.zensrViewer.staticResult));
  await page.click("#pixelated");
  await page.click("#flicker");
  const initial = await page.evaluate(() => window.zensrViewer.stats.mode);
  await page.waitForFunction(
    (mode) => window.zensrViewer.stats.mode !== mode,
    initial,
  );
  await page.click("#flicker");
  await page.screenshot({ path: out + "/gallery-detail.png", fullPage: true });
  assert.deepEqual(errors, []);
  await writeFile(
    out + "/gallery.json",
    JSON.stringify(
      { referenceCompare: true, nativePixels: true, flicker: true, href },
      null,
      2,
    ) + "\n",
  );
  console.log("Gallery links, native pixels and flicker passed.");
} finally {
  await browser.close();
}
