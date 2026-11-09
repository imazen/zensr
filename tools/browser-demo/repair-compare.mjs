// Controlled paired Rust-WASM repair calls. This is an arithmetic experiment, not IQA.
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
  const page = await browser.newPage({ ignoreHTTPSErrors: true });
  await page.goto("https://localhost:3331");
  await page.locator("#file").setInputFiles(input);
  await page.evaluate(async () => {
    const strict = await import("./repair/zensr_browser_repair.js"),
      relaxed = await import("./repair-relaxed/zensr_browser_repair.js");
    await strict.default();
    await relaxed.default();
    window.repairExperiment = {
      strict,
      relaxed,
      jpeg: new Uint8Array(
        await document.getElementById("file").files[0].arrayBuffer(),
      ),
      weights: new Uint8Array(
        await (await fetch("./models/canon4.raw")).arrayBuffer(),
      ),
    };
  });
  const cdp = await page.context().newCDPSession(page);
  await cdp.send("Profiler.enable");
  await cdp.send("Profiler.setSamplingInterval", { interval: 100 });
  await cdp.send("Profiler.start");
  const result = await page.evaluate(async () => {
    const { strict, relaxed, jpeg, weights } = window.repairExperiment;
    const records = [];
    for (let iteration = 0; iteration < 3; iteration++) {
      const outputs = [];
      for (const [mode, module] of [
        ["strict", strict],
        ["relaxed", relaxed],
      ]) {
        const start = performance.now(),
          result = module.repair(jpeg, weights),
          seconds = (performance.now() - start) / 1000;
        const report = {
            mode,
            seconds,
            width: result.width,
            height: result.height,
            applied: result.applied,
          },
          rgb = result.take_pixels();
        report.sha256 = [
          ...new Uint8Array(await crypto.subtle.digest("SHA-256", rgb)),
        ]
          .map((x) => x.toString(16).padStart(2, "0"))
          .join("");
        outputs.push({ report, rgb });
      }
      let different = 0,
        max = 0;
      for (let i = 0; i < outputs[0].rgb.length; i++) {
        const d = Math.abs(outputs[0].rgb[i] - outputs[1].rgb[i]);
        if (d) different++;
        max = Math.max(max, d);
      }
      records.push({
        iteration,
        strict: outputs[0].report,
        relaxed: outputs[1].report,
        different,
        max,
        values: outputs[0].rgb.length,
      });
    }
    return records;
  });
  const { profile } = await cdp.send("Profiler.stop");
  await writeFile(out + "/cpu-profile.json", JSON.stringify(profile));
  await writeFile(
    out + "/comparison.json",
    JSON.stringify(result, null, 2) + "\n",
  );
  console.log(result);
  assert.ok(
    result.every((row) => row.different === 0),
    "Relaxed RGB8 must match this strict reference before promotion",
  );
} finally {
  await browser.close();
}
