//! Verify qtables coverage against the survey TSV.
use std::io::BufRead;
fn main() {
    let path = std::env::args().nth(1).expect("tables tsv");
    let tol: u16 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let f = std::io::BufReader::new(std::fs::File::open(&path).unwrap());
    let (mut n, mut exact, mut tolm, mut ps, mut unk, mut encm) =
        (0u64, 0u64, 0u64, 0u64, 0u64, 0u64);
    for (i, line) in f.lines().enumerate() {
        let line = line.unwrap();
        if i == 0 {
            continue;
        }
        let c: Vec<&str> = line.split('\t').collect();
        if c.len() < 8 || c[1] != "0" {
            continue;
        }
        let v: Vec<u16> = c[7].split(',').filter_map(|x| x.parse().ok()).collect();
        if v.len() != 64 {
            continue;
        }
        let mut t = [0u16; 64];
        t.copy_from_slice(&v);
        n += 1;
        match zensr_zenjpeg::qtables::identify_luma(&t, tol) {
            zensr_zenjpeg::qtables::TableId::Preset { exact: true, .. } => exact += 1,
            zensr_zenjpeg::qtables::TableId::Preset { exact: false, .. } => tolm += 1,
            zensr_zenjpeg::qtables::TableId::Photoshop { .. } => ps += 1,
            zensr_zenjpeg::qtables::TableId::Encoder { .. } => encm += 1,
            zensr_zenjpeg::qtables::TableId::Unrecognised => unk += 1,
            _ => unk += 1, // TableId is #[non_exhaustive]
        }
    }
    let id = exact + tolm + ps + encm;
    println!("luma tables scanned: {n}");
    println!(
        "  preset exact      {exact:6} ({:.1}%)",
        100.0 * exact as f64 / n as f64
    );
    println!(
        "  preset tol<={tol}      {tolm:6} ({:.1}%)",
        100.0 * tolm as f64 / n as f64
    );
    println!(
        "  photoshop         {ps:6} ({:.1}%)",
        100.0 * ps as f64 / n as f64
    );
    println!(
        "  UNRECOGNISED      {unk:6} ({:.1}%)",
        100.0 * unk as f64 / n as f64
    );
    println!(
        "  identified total  {id:6} ({:.1}%)",
        100.0 * id as f64 / n as f64
    );
}
