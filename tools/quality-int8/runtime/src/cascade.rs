//! Four-arm SR composition helper using the public default repair API.
use super::{decode, hash, measure, planes, reply, resize, retain_png, rgb8, text};
use rgb::FromSlice;
use serde_json::{Value, json};
use std::{
    fs,
    io::{self, BufRead, Read, Write},
    path::Path,
};
use zensr_micro::{SpanfModel, adopted::AdoptedModel};
use zensr_zenjpeg::api::{Budget, Outcome, Restorer};

fn bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn repair(
    restorer: &Restorer,
    jpeg: Option<&[u8]>,
    low: &[f32],
    w: usize,
    h: usize,
) -> (Vec<f32>, Value) {
    let Some(jpeg) = jpeg else {
        return (
            low.to_vec(),
            json!({"applied":false,"reason":"clean passthrough","consistency":null}),
        );
    };
    let result = restorer
        .restore(jpeg)
        .budget(Budget::Realtime)
        .threads(1)
        .run()
        .unwrap();
    assert_eq!((result.pixels().width(), result.pixels().height()), (w, h));
    let repaired = result.pixels().planes().to_vec();
    assert!(repaired.iter().all(|v| v.is_finite()));
    let applied = matches!(&result, Outcome::Restored { .. });
    if !applied {
        assert_eq!(
            repaired, low,
            "routed-out f32 pixels differ from plain decode"
        );
    }
    let (reason, consistency) = match &result {
        Outcome::Restored { consistency, .. } => {
            ("restored".to_string(), Some(format!("{consistency:?}")))
        }
        Outcome::Unchanged { why, .. } => (format!("{why:?}"), None),
    };
    let report = result.report();
    (
        repaired,
        json!({"applied":applied,"reason":reason,"consistency":consistency,
        "quality":report.quality,"quality_scale":report.quality_scale,"chroma":format!("{:?}",report.chroma),
        "deblocked":report.deblocked,"clamped_fraction":report.clamped_fraction}),
    )
}

pub(super) fn run(args: &[String]) {
    assert_eq!(args.len(), 5, "cascade canon4-f16 spanf-f16 artifacts");
    let canon_bytes = fs::read(&args[2]).unwrap();
    let spanf_bytes = fs::read(&args[3]).unwrap();
    let artifacts = Path::new(&args[4]);
    let canon =
        AdoptedModel::load_compact(&zensr_micro::decode_all_f16(&canon_bytes), 24, 8, 1).unwrap();
    let restorer = Restorer::new().with_tier(Budget::Realtime, canon);
    let spanf = SpanfModel::from_f16_bytes(&spanf_bytes).unwrap();
    let stdin = io::stdin();
    let mut input = stdin.lock();
    let stdout = io::stdout();
    let mut output = io::BufWriter::new(stdout.lock());
    reply(
        &mut output,
        &json!({"ready":true,"canon4_sha256":hash(&canon_bytes),"spanf_sha256":hash(&spanf_bytes)}),
    );
    let mut context: Option<(Value, Vec<u8>, bool)> = None;
    loop {
        let mut line = String::new();
        if input.read_line(&mut line).unwrap() == 0 {
            break;
        }
        let request: Value = serde_json::from_str(&line).unwrap();
        match text(&request, "action") {
            "prepare" => {
                let row = &request["row"];
                let encoded = fs::read(text(row, "input")).unwrap();
                let reference = fs::read(text(row, "reference")).unwrap();
                assert_eq!(hash(&encoded), text(row, "input_sha256"));
                assert_eq!(hash(&reference), text(row, "reference_sha256"));
                let (w, h) = (
                    row["width"].as_u64().unwrap() as usize,
                    row["height"].as_u64().unwrap() as usize,
                );
                assert!(w * h <= 1_048_576 && w % 4 == 0 && h % 4 == 0);
                assert_eq!(reference.len(), 3 * w * h);
                let clean = row["q"] == 0;
                let low = if clean {
                    assert_eq!(encoded.len(), 3 * w * h / 16);
                    planes(&encoded, w / 4, h / 4)
                } else {
                    let (p, lw, lh) = decode(&encoded, false);
                    assert_eq!((lw, lh), (w / 4, h / 4));
                    p
                };
                let (repaired, mut routing) = repair(
                    &restorer,
                    if clean { None } else { Some(&encoded) },
                    &low,
                    w / 4,
                    h / 4,
                );
                let applied = routing["applied"].as_bool().unwrap();
                let preview = rgb8(&repaired, w / 4, h / 4);
                routing["preview"] = json!({"rgb8_sha256":hash(&preview),"png":retain_png(imgref::ImgRef::new(preview.as_rgb(),w/4,h/4),artifacts)});
                let direct = zensr_micro::spanf_x4_tiled(&spanf, &low, h / 4, w / 4, 1, 0);
                let restored = if applied {
                    zensr_micro::spanf_x4_tiled(&spanf, &repaired, h / 4, w / 4, 1, 0)
                } else {
                    direct.clone()
                };
                let baseline = planes(&resize(&rgb8(&low, w / 4, h / 4), w / 4, h / 4, w, h), w, h);
                let (plain_wire, repair_wire) = (bytes(&low), bytes(&repaired));
                reply(
                    &mut output,
                    &json!({"id":row["id"],"input_bytes":plain_wire.len(),
                    "plain_f32_sha256":hash(&plain_wire),"repair_f32_sha256":hash(&repair_wire),"routing":routing,
                    "baseline":measure(&reference,&baseline,w,h,artifacts),
                    "spanf_direct":measure(&reference,&direct,w,h,artifacts),
                    "spanf_repair":measure(&reference,&restored,w,h,artifacts)}),
                );
                output.write_all(&plain_wire).unwrap();
                output.write_all(&repair_wire).unwrap();
                output.flush().unwrap();
                context = Some((row.clone(), reference, applied));
            }
            "score" => {
                let (row, reference, applied) = context.as_ref().unwrap();
                assert_eq!(request["id"], row["id"]);
                let (w, h) = (
                    row["width"].as_u64().unwrap() as usize,
                    row["height"].as_u64().unwrap() as usize,
                );
                assert_eq!(request["bytes"].as_u64().unwrap() as usize, 12 * w * h);
                let mut wire = vec![0; 24 * w * h];
                input.read_exact(&mut wire).unwrap();
                if !applied {
                    assert_eq!(
                        wire[..12 * w * h],
                        wire[12 * w * h..],
                        "skipped DSCF outputs differ"
                    );
                }
                let values: Vec<f32> = wire
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| f32::from_le_bytes(*b))
                    .collect();
                let (direct, repaired) = values.split_at(3 * w * h);
                reply(
                    &mut output,
                    &json!({"id":row["id"],"dscf_direct":measure(reference,direct,w,h,artifacts),
                    "dscf_repair":measure(reference,repaired,w,h,artifacts)}),
                );
            }
            "quit" => break,
            _ => panic!("unknown cascade action"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tensor_transport_preserves_float_bits() {
        let values = [0.0, 1.0, -0.0001, f32::from_bits(1), 0.41317];
        let wire = bytes(&values);
        let roundtrip: Vec<_> = wire
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        assert_eq!(
            values.map(f32::to_bits).as_slice(),
            roundtrip.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }
}
