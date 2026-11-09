// Exercise real WASM modules, failed relaxed compilation, and independent 1× output.
import assert from "node:assert/strict";
import { mkdir, writeFile, readFile } from "node:fs/promises";
import { chromium } from "playwright";
const [input, profiled, out] = process.argv.slice(2);
await mkdir(out, { recursive: true });
const browser = await chromium.launch({
  executablePath: process.env.ZENSR_CHROME,
  headless: true,
  args: ["--no-sandbox", "--use-angle=swiftshader"],
});
try {
  const records = [];
  for (const mode of [
    "relaxed",
    "unfused",
    "blob",
    "unavailable",
    "profiled",
    "bad-icc",
  ]) {
    const context = await browser.newContext({
      ignoreHTTPSErrors: true,
      acceptDownloads: true,
    });
    if (mode === "blob")
      await context.addInitScript(() => {
        Object.defineProperty(StorageManager.prototype, "getDirectory", {
          value: undefined,
        });
      });
    if (mode === "unfused" || mode === "unavailable")
      await context.route("**/repair-relaxed/*_bg.wasm", (route) =>
        route.fulfill({
          status: 200,
          contentType: "application/wasm",
          body: Buffer.from("invalid relaxed WASM witness"),
        }),
      );
    if (mode === "unavailable")
      await context.route("**/repair-unfused/*_bg.wasm", (route) =>
        route.fulfill({
          status: 200,
          contentType: "application/wasm",
          body: Buffer.from("invalid unfused WASM witness"),
        }),
      );
    const page = await context.newPage(),
      requests = [];
    page.on("request", (request) =>
      requests.push({ method: request.method(), url: request.url() }),
    );
    await page.goto("https://localhost:3331");
    await page.waitForFunction(
      () => !document.getElementById("status").textContent.includes("Checking"),
    );
    // PNG is ineligible for the JPEG-only action.
    await page.locator("#file").setInputFiles({
      name: "ineligible.png",
      mimeType: "image/png",
      buffer: Buffer.from("header only"),
    });
    assert.equal(await page.locator("#repair-only").isDisabled(), true);
    if (mode === "bad-icc") {
      const jpeg = await readFile(input),
        payload = Buffer.concat([
          Buffer.from("ICC_PROFILE\0", "binary"),
          Buffer.from([1, 1]),
          Buffer.from("invalid profile witness"),
        ]);
      const length = Buffer.alloc(2);
      length.writeUInt16BE(payload.length + 2);
      await page.locator("#file").setInputFiles({
        name: "invalid-profile.jpg",
        mimeType: "image/jpeg",
        buffer: Buffer.concat([
          jpeg.subarray(0, 2),
          Buffer.from([255, 226]),
          length,
          payload,
          jpeg.subarray(2),
        ]),
      });
    } else
      await page
        .locator("#file")
        .setInputFiles(mode === "profiled" ? profiled : input);
    await page.selectOption("#scale", "4");
    await page.selectOption("#model", "dscf");
    assert.equal(
      await page.locator("#repair").isChecked(),
      false,
      "Standalone repair must not depend on the SR checkbox",
    );
    await page.click("#repair-only");
    await page.waitForFunction(
      () => window.zensrState.done || window.zensrState.error,
      {},
      { timeout: 60000 },
    );
    const result = await page.evaluate(() => window.zensrState);
    assert.ok(requests.every((r) => r.method === "GET"));
    assert.ok(
      !requests.some((r) => /\.onnx|ort.*\.mjs|\/repair\/zensr/.test(r.url)),
      "Repair-only must not load an SR model/runtime or strict fused reference",
    );
    if (mode === "unavailable" || mode === "bad-icc") {
      assert.equal(result.done, false);
      assert.match(
        result.error,
        mode === "bad-icc"
          ? /[Ii][Cc][Cc]|profile|[Cc]olor/
          : /relaxed.*unfused/,
      );
      assert.equal(await page.locator("#download").isHidden(), true);
      if (mode === "bad-icc")
        assert.ok(
          !requests.some((r) => r.url.includes("repair-unfused")),
          "Input errors must not be retried with another arithmetic mode",
        );
    } else {
      assert.equal(result.error, null);
      assert.equal(result.stats.operation, "repair");
      assert.equal(result.stats.scale, 1);
      assert.equal(result.stats.model, "canon4");
      assert.equal(
        result.stats.repair.math,
        ["blob", "profiled"].includes(mode) ? "relaxed" : mode,
      );
      assert.ok(result.stats.repair.applied);
      assert.deepEqual(
        [result.stats.outputWidth, result.stats.outputHeight],
        [result.stats.width, result.stats.height],
      );
      assert.equal(result.stats.maxTileFloats, 0);
      assert.equal(
        result.checkpoint,
        null,
        "Standalone repair is the final output, not a duplicated SR checkpoint",
      );
      const download = page.waitForEvent("download");
      await page.click("#download");
      await (await download).saveAs(out + "/" + mode + ".png");
      if (mode === "blob") {
        assert.equal(
          await page.evaluate(() => window.zensrViewer.stats.detail),
          false,
        );
      } else {
        await page.click("#zoom-native");
        await page.waitForFunction(
          () =>
            window.zensrViewer.stats.nativeTiles > 0 &&
            window.zensrViewer.stats.loading === 0,
        );
        assert.equal(
          await page.evaluate(() => window.zensrViewer.stats.detail),
          true,
        );
        await page.reload();
        await page.waitForFunction(
          () => document.querySelectorAll("#saved a").length === 1,
        );
      }
    }
    const deadline = Date.now() + 5000;
    while (page.workers().length) {
      if (Date.now() > deadline)
        throw new Error("Worker remained after repair completion/failure");
      await page.waitForTimeout(20);
    }
    records.push({ mode, ...result, requests });
    await context.close();
  }
  await writeFile(
    out + "/results.json",
    JSON.stringify(records, null, 2) + "\n",
  );
  console.log(
    records.map(({ mode, done, error, stats }) => ({
      mode,
      done,
      error,
      math: stats?.repair.math,
      seconds: stats?.processingSeconds,
    })),
  );
  // Independent consumer process for raw arithmetic output; PNG readback uses Pillow separately.
  for (const mode of ["relaxed", "unfused"]) {
    const packagePath =
      mode === "relaxed"
        ? new URL(
            "./dist/repair-relaxed/zensr_browser_repair.js",
            import.meta.url,
          )
        : new URL(
            "./dist/repair-unfused/zensr_browser_repair.js",
            import.meta.url,
          );
    const module = await import(packagePath.href);
    await module.default({
      module_or_path: await readFile(
        new URL("./zensr_browser_repair_bg.wasm", packagePath),
      ),
    });
    const result = module.repair(
      await readFile(input),
      await readFile(new URL("./dist/models/canon4.raw", import.meta.url)),
    );
    await writeFile(out + "/" + mode + ".rgb8", result.take_pixels());
  }
} finally {
  await browser.close();
}
