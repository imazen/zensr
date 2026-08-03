//! List which minted tables the identifier does NOT yet know, with attribution.
use std::io::BufRead;
fn main() {
    let path = std::env::args().nth(1).expect("tsv");
    let f = std::io::BufReader::new(std::fs::File::open(&path).unwrap());
    for (i, line) in f.lines().enumerate() {
        let line = line.unwrap();
        if i == 0 {
            continue;
        }
        let c: Vec<&str> = line.split('\t').collect();
        // dest 0 only: these are LUMA tables, and the identifier
        // carries luma sets. Scoring chroma against them is meaningless.
        if c.len() < 8 || c[1] != "0" {
            continue;
        }
        let v: Vec<u16> = c[7].split(',').filter_map(|x| x.parse().ok()).collect();
        if v.len() != 64 {
            continue;
        }
        let mut t = [0u16; 64];
        t.copy_from_slice(&v);
        if matches!(
            zensr_zenjpeg::qtables::identify_luma(&t, 1),
            zensr_zenjpeg::qtables::TableId::Unrecognised
        ) {
            println!("{}\t{}\t{}", c[0], c[3], c[7]);
        }
    }
}
