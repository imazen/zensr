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
    viewport: { width: 1450, height: 1000 },
  });
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("https://localhost:3331");
  await page.waitForFunction(
    () => !document.getElementById("status").textContent.includes("Checking"),
  );
  await page.locator("#file").setInputFiles(input);
  await page.selectOption("#backend", "wasm");
  await page.selectOption("#cpu-workers", "4");
  await page.selectOption("#scale", "4");
  if (await page.locator("#repair").isEnabled())
    await page.locator("#repair").uncheck();
  await page.click("#start");
  await page.waitForFunction(
    () => window.zensrState.done || window.zensrState.error,
  );
  assert.equal(await page.evaluate(() => window.zensrState.error), null);
  await page.click("#zoom-native");
  await page.waitForFunction(
    () =>
      window.zensrViewer.stats.nativeTiles > 0 &&
      window.zensrViewer.stats.loading === 0,
  );
  assert.equal(await page.evaluate(() => window.zensrViewer.stats.zoom), 1);
  const old = await page.evaluate(() => ({
    x: window.zensrViewer.ox,
    y: window.zensrViewer.oy,
  }));
  const box = await page.locator("#display").boundingBox();
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.down();
  await page.mouse.move(
    box.x + box.width / 2 + 63,
    box.y + box.height / 2 + 27,
  );
  await page.mouse.up();
  const moved = await page.evaluate(() => ({
    x: window.zensrViewer.ox,
    y: window.zensrViewer.oy,
  }));
  assert.equal(moved.x - old.x, 63);
  assert.equal(moved.y - old.y, 27);
  await page.click("#pixelated");
  assert.equal(
    await page.evaluate(
      () =>
        document.getElementById("display").getContext("2d")
          .imageSmoothingEnabled,
    ),
    false,
  );
  await page.click("#zoom-in");
  await page.click("#zoom-in");
  assert.equal(await page.evaluate(() => window.zensrViewer.stats.zoom), 4);
  await page.locator("#display").focus();
  await page.keyboard.down("Space");
  assert.equal(
    await page.evaluate(() => window.zensrViewer.stats.mode),
    "original",
  );
  await page.keyboard.up("Space");
  await page.click("#flicker");
  const first = await page.evaluate(() => window.zensrViewer.stats.mode);
  await page.waitForFunction(
    (value) => window.zensrViewer.stats.mode !== value,
    first,
  );
  await page.click("#flicker");
  await page.waitForFunction(() => window.zensrViewer.stats.loading === 0);
  await page.screenshot({ path: out + "/native-detail.png", fullPage: true });
  await page.click("#zoom-fit");
  await page.screenshot({ path: out + "/fit.png", fullPage: true });
  const stats = await page.evaluate(() => window.zensrViewer.stats);
  assert.ok(stats.cacheBytes <= 32 * 1048576);
  assert.deepEqual(errors, []);
  // Retain exact encoded native-detail witnesses for independent PNG/crop validation.
  const tiles = await page.evaluate(async () => {
    const root = await navigator.storage.getDirectory(),
      folder = await root.getDirectoryHandle(
        window.zensrState.stats.outputName + ".tiles",
      ),
      result = [];
    for await (const [name, handle] of folder.entries())
      result.push({
        name,
        bytes: [
          ...new Uint8Array(await (await handle.getFile()).arrayBuffer()),
        ],
      });
    return result;
  });
  for (const tile of tiles)
    await writeFile(out + "/" + tile.name, Buffer.from(tile.bytes));
  await writeFile(
    out + "/viewer.json",
    JSON.stringify(
      {
        stats,
        panDelta: { x: 63, y: 27 },
        flicker: true,
        spaceHold: true,
        pixelated: true,
        tiles: tiles.length,
      },
      null,
      2,
    ) + "\n",
  );
  console.log({ stats, tiles: tiles.length });
} finally {
  await browser.close();
}
