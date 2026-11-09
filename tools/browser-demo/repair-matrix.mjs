// Paired arithmetic checks on an explicit, hash-identified JPEG cohort.
import assert from "node:assert/strict";
import { readFile, writeFile, mkdir } from "node:fs/promises";
import { createHash } from "node:crypto";
import { chromium } from "playwright";
const [manifest, out] = process.argv.slice(2);
const cases = JSON.parse(await readFile(manifest, "utf8"));
assert.ok(cases.length > 0);
await mkdir(out, { recursive: true });
const browser = await chromium.launch({
  executablePath: process.env.ZENSR_CHROME,
  headless: true,
  args: ["--no-sandbox"],
});
try {
  const page = await browser.newPage({ ignoreHTTPSErrors: true });
  await page.goto("https://localhost:3331");
  await page.evaluate(async () => {
    const strict = await import("./repair/zensr_browser_repair.js"),
      relaxed = await import("./repair-relaxed/zensr_browser_repair.js");
    await strict.default();
    await relaxed.default();
    window.mathModules = { strict, relaxed };
    window.weights = new Uint8Array(
      await (await fetch("./models/canon4.raw")).arrayBuffer(),
    );
  });
  const records = [];
  for (const row of cases) {
    const bytes = await readFile(row.input),
      sha = createHash("sha256").update(bytes).digest("hex");
    assert.equal(sha, row.input_sha256);
    const result = await page.evaluate(
      ({ bytes }) => {
        const jpeg = new Uint8Array(bytes),
          outputs = [];
        for (const mode of ["strict", "relaxed"]) {
          const module = window.mathModules[mode],
            start = performance.now(),
            result = module.repair(jpeg, window.weights),
            seconds = (performance.now() - start) / 1000;
          outputs.push({
            mode,
            seconds,
            width: result.width,
            height: result.height,
            applied: result.applied,
            rgb: result.take_pixels(),
          });
        }
        const [a, b] = outputs;
        let different = 0,
          max = 0;
        for (let i = 0; i < a.rgb.length; i++) {
          const d = Math.abs(a.rgb[i] - b.rgb[i]);
          if (d) different++;
          max = Math.max(max, d);
        }
        const values = a.rgb.length;
        return {
          different,
          max,
          values,
          strict: { ...a, rgb: undefined },
          relaxed: { ...b, rgb: undefined },
        };
      },
      { bytes: [...bytes] },
    );
    records.push({ ...row, ...result });
    await writeFile(
      out + "/matrix.json",
      JSON.stringify(records, null, 2) + "\n",
    );
    assert.deepEqual(
      [result.relaxed.width, result.relaxed.height, result.relaxed.applied],
      [result.strict.width, result.strict.height, result.strict.applied],
    );
  }
  console.log({
    cases: records.length,
    applied: records.filter((r) => r.strict.applied).length,
    values: records.reduce((n, r) => n + r.values, 0),
    different: records.reduce((n, r) => n + r.different, 0),
    max: Math.max(...records.map((r) => r.max)),
  });
  assert.ok(
    records.every((r) => r.different === 0),
    "Relaxed RGB8 differs from strict; inspect the saved matrix before adoption",
  );
} finally {
  await browser.close();
}
