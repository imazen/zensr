// JPEG pixels, including the viewer's original, come from zenjpeg with ICC→sRGB.
self.onmessage = async ({ data: { port, jpeg } }) => {
  try {
    const codec = await import("./jpeg/zensr_browser_jpeg_probe.js");
    await codec.default();
    const decoded = codec.decode(jpeg),
      width = decoded.width,
      height = decoded.height;
    const rgb = decoded.take_pixels(),
      wasmBytes = codec.memory_bytes();
    port.postMessage({ width, height, rgb, jpeg, wasmBytes }, [
      rgb.buffer,
      jpeg.buffer,
    ]);
  } catch (error) {
    port.postMessage({
      error: `zenjpeg decode failed: ${error.message || error}`,
    });
  }
};
