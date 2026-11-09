// Fallback is for module loading/initialization; data errors remain data errors.
export async function repairRuntime(load = loadModule) {
  let fallbackReason;
  try {
    const codec = await load("relaxed");
    return { codec, math: "relaxed", fallbackReason: null };
  } catch (error) {
    fallbackReason = String(error.message || error);
  }
  try {
    const codec = await load("unfused");
    return { codec, math: "unfused", fallbackReason };
  } catch (error) {
    throw new Error(
      `JPEG repair could not initialize relaxed SIMD (${fallbackReason}) or unfused SIMD (${error.message || error}).`,
    );
  }
}
async function loadModule(math) {
  const codec = await import(
    math === "relaxed"
      ? "./repair-relaxed/zensr_browser_repair.js"
      : "./repair-unfused/zensr_browser_repair.js"
  );
  await codec.default();
  return codec;
}
