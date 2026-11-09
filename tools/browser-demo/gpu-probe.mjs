import { chromium } from "playwright";
if (!process.env.ZENSR_CHROME)
  throw new Error("Set ZENSR_CHROME to the tested Chromium executable.");
const browser = await chromium.launch({
  executablePath: process.env.ZENSR_CHROME,
  headless: true,
  dumpio: true,
  args: [
    "--no-sandbox",
    "--enable-gpu",
    "--enable-unsafe-webgpu",
    "--ignore-gpu-blocklist",
    "--use-angle=vulkan",
    "--use-vulkan=native",
    "--enable-features=Vulkan",
    "--disable-vulkan-surface",
    "--enable-dawn-features=allow_unsafe_apis",
  ],
});
const page = await browser.newPage();
await page.goto(process.env.ZENSR_URL || "http://localhost:3330");
console.log(
  await page.evaluate(async () => {
    const a = await navigator.gpu?.requestAdapter({
      powerPreference: "high-performance",
    });
    return {
      secure: isSecureContext,
      adapter: !!a,
      info: a
        ? {
            vendor: a.info.vendor,
            architecture: a.info.architecture,
            description: a.info.description,
          }
        : null,
    };
  }),
);
const cdp = await browser.newBrowserCDPSession();
const { gpu } = await cdp.send("SystemInfo.getInfo");
console.log({ features: gpu.featureStatus });
await browser.close();
