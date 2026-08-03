//! Concurrent quantization-table survey over a large JPEG corpus.
//!
//! Built to expand encoder / quant-table recognition: the probe currently
//! reports `Unknown` for real-world CDN traffic, which silently falls back to
//! the round-to-nearest projection slack (0.15). Knowing which tables actually
//! occur in the wild — and how they cluster — is what fixes that.
//!
//! Two things make this fast on a high-latency mount:
//!   * only the HEADER is read (default 64 KiB), not the whole file. DQT/SOF
//!     live before SOS, so a 6 MB JPEG costs the same as a 20 KB one.
//!   * the pool is deliberately oversubscribed. This is IO-latency-bound, not
//!     CPU-bound, so threads should outnumber cores several times over.
//!
//! Outputs two TSVs so the join stays cheap: one row per file, one row per
//! (file, table) with all 64 coefficients in natural (de-zigzagged) order.
//!
//! Usage: dqt_survey <root>... --files <tsv> --tables <tsv> [--threads N] [--head-kb N]

use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;

const ZIGZAG: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

struct Header {
    tables: Vec<(u8, u8, [u16; 64])>, // (dest_id, precision, natural-order values)
    width: u32,
    height: u32,
    ncomp: u8,
    /// (h_samp, v_samp) per component, in declaration order.
    samp: Vec<(u8, u8)>,
    progressive: bool,
    saw_sos: bool,
    app_markers: Vec<u8>,
}

/// Parse just enough of the header. Returns None only if this is not a JPEG.
fn parse_header(b: &[u8]) -> Option<Header> {
    if b.len() < 4 || b[0] != 0xFF || b[1] != 0xD8 {
        return None;
    }
    let mut h = Header {
        tables: Vec::new(),
        width: 0,
        height: 0,
        ncomp: 0,
        samp: Vec::new(),
        progressive: false,
        saw_sos: false,
        app_markers: Vec::new(),
    };
    let mut i = 2usize;
    while i + 3 < b.len() {
        if b[i] != 0xFF {
            i += 1;
            continue;
        }
        let m = b[i + 1];
        // standalone markers carry no length
        if m == 0xD8 || m == 0x01 || (0xD0..=0xD7).contains(&m) {
            i += 2;
            continue;
        }
        if m == 0xD9 {
            break;
        }
        let len = ((b[i + 2] as usize) << 8) | b[i + 3] as usize;
        if len < 2 || i + 2 + len > b.len() {
            break; // truncated header — report what we have
        }
        let seg = &b[i + 4..i + 2 + len];
        match m {
            0xDB => {
                let mut j = 0usize;
                while j < seg.len() {
                    let pq = seg[j] >> 4;
                    let tq = seg[j] & 0x0F;
                    j += 1;
                    let n = if pq == 0 { 64 } else { 128 };
                    if j + n > seg.len() {
                        break;
                    }
                    let mut nat = [0u16; 64];
                    for k in 0..64 {
                        let v = if pq == 0 {
                            seg[j + k] as u16
                        } else {
                            ((seg[j + k * 2] as u16) << 8) | seg[j + k * 2 + 1] as u16
                        };
                        nat[ZIGZAG[k]] = v;
                    }
                    h.tables.push((tq, pq, nat));
                    j += n;
                }
            }
            0xC0 | 0xC1 | 0xC2 | 0xC3 | 0xC5 | 0xC6 | 0xC7 | 0xC9 | 0xCA | 0xCB | 0xCD | 0xCE
            | 0xCF => {
                h.progressive = m == 0xC2 || m == 0xC6 || m == 0xCA || m == 0xCE;
                if seg.len() >= 6 {
                    h.height = ((seg[1] as u32) << 8) | seg[2] as u32;
                    h.width = ((seg[3] as u32) << 8) | seg[4] as u32;
                    h.ncomp = seg[5];
                    for c in 0..h.ncomp as usize {
                        let o = 6 + c * 3;
                        if o + 1 < seg.len() {
                            h.samp.push((seg[o + 1] >> 4, seg[o + 1] & 0x0F));
                        }
                    }
                }
            }
            0xDA => {
                h.saw_sos = true;
                return Some(h);
            }
            0xE0..=0xEF => h.app_markers.push(m - 0xE0),
            _ => {}
        }
        i += 2 + len;
    }
    Some(h)
}

fn subsampling(samp: &[(u8, u8)]) -> String {
    match samp {
        [(1, 1), (1, 1), (1, 1)] => "444".into(),
        [(2, 2), (1, 1), (1, 1)] => "420".into(),
        [(2, 1), (1, 1), (1, 1)] => "422".into(),
        [(1, 2), (1, 1), (1, 1)] => "440".into(),
        [(1, 1)] => "gray".into(),
        s => s
            .iter()
            .map(|(a, b)| format!("{a}x{b}"))
            .collect::<Vec<_>>()
            .join(","),
    }
}

fn fnv(vals: &[u16]) -> String {
    let mut h: u64 = 0xcbf29ce484222325;
    for &v in vals {
        for b in v.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    format!("{h:016x}")
}

/// Directory enumeration is itself latency-bound on this mount — a serial walk
/// of the tree outran the header reads it was feeding. Subdirectories are
/// therefore walked in parallel, so enumeration proceeds at the pool's width
/// instead of one stat at a time.
fn walk(p: &Path, excl: &[String]) -> Vec<PathBuf> {
    use rayon::prelude::*;
    // Skipping non-corpus trees is not a nicety: 63 git clones under
    // _repo_clones contribute tens of thousands of object directories, each a
    // round trip on this mount, and zero images. They dominated the walk.
    if let Some(name) = p.file_name().and_then(|n| n.to_str()) {
        if excl.iter().any(|e| e == name) {
            return Vec::new();
        }
    }
    let Ok(rd) = std::fs::read_dir(p) else {
        return Vec::new();
    };
    let (mut dirs, mut files) = (Vec::new(), Vec::new());
    for e in rd.flatten() {
        let path = e.path();
        match e.file_type() {
            Ok(t) if t.is_dir() => dirs.push(path),
            Ok(t) if t.is_file() => {
                let n = path.to_string_lossy().to_ascii_lowercase();
                if n.ends_with(".jpg") || n.ends_with(".jpeg") || n.ends_with(".jpe") {
                    files.push(path);
                }
            }
            _ => {}
        }
    }
    let sub: Vec<PathBuf> = dirs.par_iter().flat_map(|d| walk(d, excl)).collect();
    files.extend(sub);
    files
}

fn arg(name: &str, def: &str) -> String {
    let a: Vec<String> = std::env::args().collect();
    a.iter()
        .position(|x| x == name)
        .and_then(|i| a.get(i + 1).cloned())
        .unwrap_or_else(|| def.to_string())
}

fn main() {
    let roots: Vec<String> = std::env::args()
        .skip(1)
        .take_while(|a| !a.starts_with("--"))
        .collect();
    if roots.is_empty() {
        eprintln!(
            "usage: dqt_survey <root>... --files <tsv> --tables <tsv> [--threads N] [--head-kb N]"
        );
        std::process::exit(2);
    }
    let files_out = arg("--files", "dqt_files.tsv");
    let tables_out = arg("--tables", "dqt_tables.tsv");
    let threads: usize = arg("--threads", "96").parse().unwrap_or(96);
    let head: usize = arg("--head-kb", "64").parse().unwrap_or(64) * 1024;

    // The pool must exist before the walk, because the walk is parallel too.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("thread pool");
    eprint!("walking (parallel)...");
    let t0 = std::time::Instant::now();
    let excl: Vec<String> = arg(
        "--exclude",
        "_repo_clones,cc-index,.git,node_modules,target",
    )
    .split(',')
    .filter(|x| !x.is_empty())
    .map(|x| x.to_string())
    .collect();
    let files: Vec<PathBuf> = pool.install(|| {
        roots
            .iter()
            .flat_map(|r| walk(Path::new(r), &excl))
            .collect()
    });
    eprintln!(
        " {} JPEG(s) in {:.1}s",
        files.len(),
        t0.elapsed().as_secs_f64()
    );
    if files.is_empty() {
        return;
    }

    let done = AtomicUsize::new(0);
    let total = files.len();
    let (tx, rx) = mpsc::channel::<(String, String)>();

    // Writer thread keeps the workers free of IO contention on the outputs.
    let writer = std::thread::spawn(move || {
        use std::io::Write;
        let mut f = std::io::BufWriter::new(std::fs::File::create(&files_out).unwrap());
        let mut t = std::io::BufWriter::new(std::fs::File::create(&tables_out).unwrap());
        writeln!(f, "path\tbytes\tw\th\tncomp\tss\tmode\tapp\tn_tables\tluma_hash\tchroma_hash\tprobe_family\tprobe_q\tprobe_scale\ttruncated").unwrap();
        writeln!(t, "path\tdest\tprec\thash\tdc\tmean\tmax\tvalues").unwrap();
        for (a, b) in rx {
            if !a.is_empty() {
                f.write_all(a.as_bytes()).unwrap();
            }
            if !b.is_empty() {
                t.write_all(b.as_bytes()).unwrap();
            }
        }
    });

    // IO-latency-bound: oversubscribe hard rather than matching core count.
    pool.install(|| {
        use rayon::prelude::*;
        files.par_iter().for_each_with(tx.clone(), |tx, p| {
            let mut buf = vec![0u8; head];
            let n = match std::fs::File::open(p).and_then(|mut fh| fh.read(&mut buf)) {
                Ok(n) => n,
                Err(_) => return,
            };
            buf.truncate(n);
            let size = std::fs::metadata(p).map(|m| m.len()).unwrap_or(0);
            let Some(h) = parse_header(&buf) else { return };
            let path = p.to_string_lossy().replace('\t', " ");

            let (mut luma_hash, mut chroma_hash) = (String::new(), String::new());
            let mut tbl_rows = String::new();
            for (dest, prec, vals) in &h.tables {
                let hash = fnv(vals);
                if *dest == 0 && luma_hash.is_empty() {
                    luma_hash = hash.clone();
                } else if *dest != 0 && chroma_hash.is_empty() {
                    chroma_hash = hash.clone();
                }
                let sum: u32 = vals.iter().map(|&v| v as u32).sum();
                let mx = vals.iter().copied().max().unwrap_or(0);
                let joined: Vec<String> = vals.iter().map(|v| v.to_string()).collect();
                tbl_rows.push_str(&format!(
                    "{path}\t{dest}\t{prec}\t{hash}\t{}\t{:.2}\t{mx}\t{}\n",
                    vals[0],
                    sum as f64 / 64.0,
                    joined.join(",")
                ));
            }

            // The probe is the thing we are trying to improve, so record what it
            // says today — on the same header bytes, no extra IO.
            let (fam, q, scale) = match zenjpeg::detect::probe(&buf) {
                Ok(pr) => (
                    format!("{:?}", pr.encoder),
                    format!("{:.1}", pr.quality.value),
                    format!("{:?}", pr.quality.scale),
                ),
                Err(_) => ("ProbeErr".into(), "".into(), "".into()),
            };
            let apps: Vec<String> = h.app_markers.iter().map(|a| a.to_string()).collect();
            let file_row = format!(
                "{path}\t{size}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{luma_hash}\t{chroma_hash}\t{fam}\t{q}\t{scale}\t{}\n",
                h.width,
                h.height,
                h.ncomp,
                subsampling(&h.samp),
                if h.progressive { "progressive" } else { "baseline" },
                if apps.is_empty() { "-".into() } else { apps.join(",") },
                h.tables.len(),
                !h.saw_sos,
            );
            let _ = tx.send((file_row, tbl_rows));
            let d = done.fetch_add(1, Ordering::Relaxed) + 1;
            if d % 5000 == 0 {
                eprintln!("  {d}/{total}");
            }
        });
    });
    drop(tx);
    writer.join().unwrap();
    eprintln!("done {} files", done.load(Ordering::Relaxed));
}
