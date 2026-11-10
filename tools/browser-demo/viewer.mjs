const CACHE_LIMIT = 32 * 1048576,
  MAX_VISIBLE = 24;
export class DetailViewer {
  constructor() {
    this.canvas = document.getElementById("display");
    this.viewport = document.getElementById("viewer");
    this.preview = document.getElementById("after");
    this.mode = document.getElementById("view-mode");
    this.cache = new Map();
    this.available = new Set();
    this.loading = new Set();
    this.cacheBytes = 0;
    this.generation = 0;
    this.zoom = 1;
    this.ox = 0;
    this.oy = 0;
    this.outputWidth = 0;
    this.outputHeight = 0;
    this.stats = {
      cacheBytes: 0,
      nativeTiles: 0,
      loading: 0,
      zoom: 1,
      detail: false,
    };
    const $ = (id) => document.getElementById(id);
    $("zoom-in").onclick = () => this.zoomAt(this.zoom * 2);
    $("zoom-out").onclick = () => this.zoomAt(this.zoom / 2);
    $("zoom-fit").onclick = () => this.fit();
    $("zoom-native").onclick = () => this.zoomAt(1);
    $("pixelated").onchange = () => this.render();
    this.mode.onchange = () => this.render();
    $("wipe").oninput = () => this.render();
    $("flicker").onclick = () => {
      if (this.timer) {
        clearInterval(this.timer);
        this.timer = null;
        this.flickerOriginal = false;
      } else {
        this.flickerOriginal = true;
        this.timer = setInterval(() => {
          this.flickerOriginal = !this.flickerOriginal;
          this.render();
        }, 500);
      }
      $("flicker").setAttribute("aria-pressed", String(!!this.timer));
      this.render();
    };
    this.canvas.addEventListener(
      "wheel",
      (event) => {
        event.preventDefault();
        const box = this.canvas.getBoundingClientRect();
        this.zoomAt(
          this.zoom * Math.exp(-event.deltaY * 0.002),
          event.clientX - box.left,
          event.clientY - box.top,
        );
      },
      { passive: false },
    );
    this.canvas.addEventListener("pointerdown", (event) => {
      this.canvas.focus();
      this.drag = { x: event.clientX, y: event.clientY };
      this.canvas.setPointerCapture(event.pointerId);
    });
    this.canvas.addEventListener("pointermove", (event) => {
      if (this.drag) {
        this.ox += event.clientX - this.drag.x;
        this.oy += event.clientY - this.drag.y;
        this.drag = { x: event.clientX, y: event.clientY };
        this.render();
      }
    });
    const stop = () => {
      this.drag = null;
    };
    this.canvas.addEventListener("pointerup", stop);
    this.canvas.addEventListener("pointercancel", stop);
    this.canvas.addEventListener("keydown", (event) => {
      if (event.key === " ") {
        event.preventDefault();
        this.holdOriginal = true;
        this.render();
      }
      if (event.key === "+") this.zoomAt(this.zoom * 2);
      if (event.key === "-") this.zoomAt(this.zoom / 2);
      if (event.key === "0") this.fit();
      if (event.key === "1") this.zoomAt(1);
    });
    const release = () => {
      this.holdOriginal = false;
      this.render();
    };
    this.canvas.addEventListener("keyup", (event) => {
      if (event.key === " ") release();
    });
    this.canvas.addEventListener("blur", release);
    new ResizeObserver(() => {
      this.resize();
      if (this.fitted) this.fit();
      else this.render();
    }).observe(this.viewport);
  }
  resize() {
    const w = this.viewport.clientWidth,
      h = this.viewport.clientHeight;
    if (!w || !h) return;
    const dpr = Math.min(
      devicePixelRatio || 1,
      2,
      Math.sqrt(4_000_000 / (w * h)),
    );
    this.dpr = dpr;
    this.width = w;
    this.height = h;
    const bw = Math.round(w * dpr),
      bh = Math.round(h * dpr);
    if (this.canvas.width !== bw || this.canvas.height !== bh) {
      this.canvas.width = bw;
      this.canvas.height = bh;
    }
  }
  async input(file, metadata, directory) {
    const generation = ++this.generation;
    this.staticResult?.close();
    this.staticResult = null;
    this.source?.close();
    this.source = null;
    for (const value of this.cache.values()) value.bitmap.close();
    this.cache.clear();
    this.available.clear();
    this.cacheBytes = 0;
    this.directory = directory;
    this.outputWidth = metadata.outputWidth;
    this.outputHeight = metadata.outputHeight;
    this.core = metadata.detailCore ?? metadata.scale * 128;
    const source =
      file instanceof ImageBitmap ? file : await createImageBitmap(file);
    if (generation !== this.generation) {
      source.close();
      return;
    }
    this.source = source;
    this.resize();
    this.fit();
  }
  result(metadata, directory) {
    this.directory = directory;
    this.core = metadata.detailCore ?? metadata.scale * 128;
    this.outputWidth = metadata.outputWidth;
    this.outputHeight = metadata.outputHeight;
    this.render();
  }
  fit() {
    if (!this.outputWidth) return;
    this.resize();
    this.zoom = Math.min(
      this.width / this.outputWidth,
      this.height / this.outputHeight,
    );
    this.ox = (this.width - this.outputWidth * this.zoom) / 2;
    this.oy = (this.height - this.outputHeight * this.zoom) / 2;
    this.fitted = true;
    this.render();
  }
  zoomAt(next, x = this.width / 2, y = this.height / 2) {
    if (!this.outputWidth) return;
    const fit = Math.min(
      this.width / this.outputWidth,
      this.height / this.outputHeight,
    );
    next = Math.max(Math.min(1, fit / 4), Math.min(32, next));
    const wx = (x - this.ox) / this.zoom,
      wy = (y - this.oy) / this.zoom;
    this.zoom = next;
    this.ox = x - wx * next;
    this.oy = y - wy * next;
    this.fitted = false;
    this.render();
  }
  async tile(data) {
    if (this.directory && data.png) {
      const key = `${data.column}-${data.row}.png`,
        handle = await this.directory.getFileHandle(key, { create: true });
      const writer = await handle.createWritable();
      await writer.write(data.png);
      await writer.close();
      this.available.add(key);
    }
    this.render();
  }
  visible() {
    if (!this.core) return [];
    const left = Math.max(0, Math.floor(-this.ox / this.zoom / this.core)),
      top = Math.max(0, Math.floor(-this.oy / this.zoom / this.core));
    const right = Math.min(
      Math.ceil(this.outputWidth / this.core),
      Math.ceil((this.width - this.ox) / this.zoom / this.core),
    );
    const bottom = Math.min(
      Math.ceil(this.outputHeight / this.core),
      Math.ceil((this.height - this.oy) / this.zoom / this.core),
    );
    if ((right - left) * (bottom - top) > MAX_VISIBLE) return [];
    const keys = [];
    for (let y = top; y < bottom; y++)
      for (let x = left; x < right; x++)
        keys.push({ key: `${x}-${y}.png`, x: x * this.core, y: y * this.core });
    return keys;
  }
  render() {
    if (!this.width || !this.outputWidth || !this.source) return;
    const ctx = this.canvas.getContext("2d", { alpha: false }),
      dpr = this.dpr;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = "#e8ece1";
    ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    ctx.imageSmoothingEnabled = !document.getElementById("pixelated").checked;
    const transform = () =>
      ctx.setTransform(
        dpr * this.zoom,
        0,
        0,
        dpr * this.zoom,
        dpr * this.ox,
        dpr * this.oy,
      );
    transform();
    ctx.fillStyle = "#fff";
    ctx.fillRect(0, 0, this.outputWidth, this.outputHeight);
    ctx.drawImage(this.source, 0, 0, this.outputWidth, this.outputHeight);
    const mode =
      this.holdOriginal || this.timer
        ? this.holdOriginal || this.flickerOriginal
          ? "original"
          : "result"
        : this.mode.value;
    const visible = this.visible();
    if (mode !== "original") {
      ctx.save();
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      if (mode === "split") {
        const x =
          (this.width * Number(document.getElementById("wipe").value)) / 100;
        ctx.beginPath();
        ctx.rect(x, 0, this.width - x, this.height);
        ctx.clip();
      }
      transform();
      ctx.drawImage(
        this.staticResult || this.preview,
        0,
        0,
        this.outputWidth,
        this.outputHeight,
      );
      for (const tile of visible) {
        const entry = this.cache.get(tile.key);
        if (entry) {
          this.cache.delete(tile.key);
          this.cache.set(tile.key, entry);
          ctx.drawImage(entry.bitmap, tile.x, tile.y);
        }
      }
      ctx.restore();
    }
    document.querySelector(".label.before").hidden = mode === "result";
    document.querySelector(".label.after").hidden = mode === "original";
    document.getElementById("divider").hidden = mode !== "split";
    document.getElementById("divider").style.left =
      document.getElementById("wipe").value + "%";
    const detail =
      !!this.staticResult || (!!this.directory && visible.length > 0);
    document.getElementById("view-state").textContent =
      `${(this.zoom * 100).toFixed(0)}% · ${mode === "original" ? "original" : detail ? "native output tiles" : "reduced preview"}`;
    this.stats = {
      cacheBytes: this.cacheBytes,
      nativeTiles: this.cache.size,
      loading: this.loading.size,
      zoom: this.zoom,
      detail,
      mode,
    };
    if (detail && !this.staticResult && mode !== "original") this.load(visible);
  }
  load(visible) {
    const generation = this.generation;
    for (const tile of visible) {
      if (this.loading.size >= 2) break;
      if (
        this.cache.has(tile.key) ||
        this.loading.has(tile.key) ||
        !this.available.has(tile.key)
      )
        continue;
      this.loading.add(tile.key);
      (async () => {
        const handle = await this.directory.getFileHandle(tile.key),
          bitmap = await createImageBitmap(await handle.getFile());
        if (generation !== this.generation) {
          bitmap.close();
          return;
        }
        const bytes = bitmap.width * bitmap.height * 4;
        while (this.cacheBytes + bytes > CACHE_LIMIT && this.cache.size) {
          const [key, entry] = this.cache.entries().next().value;
          entry.bitmap.close();
          this.cache.delete(key);
          this.cacheBytes -= entry.bytes;
        }
        this.cache.set(tile.key, { bitmap, bytes });
        this.cacheBytes += bytes;
      })()
        .catch((error) => {
          this.available.delete(tile.key);
          if (generation === this.generation)
            document.getElementById("view-state").textContent =
              `Detail tile unavailable: ${error.message}`;
        })
        .finally(() => {
          this.loading.delete(tile.key);
          if (generation === this.generation) this.render();
        });
    }
  }
}
