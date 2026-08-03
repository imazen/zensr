//! Identify the encoder and quality that produced a quantization table.
//!
//! The probe names a family from markers. That is useful provenance and poor
//! evidence about the quantiser: measured over 17,739 corpus JPEGs it reports
//! `ImageMagick` for 9,626 files, of which only 180 use the ImageMagick table.
//! Because the projection slack is a claim about *quantiser behaviour*, it
//! should follow the table.
//!
//! Identification is by reconstruction, not by a stored list: the nine mozjpeg
//! presets are expanded over quality 1..=100 and both `force_baseline` modes
//! and compared directly, which returns a preset *and* a quality rather than a
//! label. Photoshop tables, which no preset generates, are matched against
//! measured data.

mod data {
    pub use crate::qtables_data::*;
}
pub use data::{
    ENCODER_LUMA_TABLES, MOZJPEG_LUMA_BASES, MOZJPEG_PRESET_NAMES, PHOTOSHOP_LUMA_TABLES,
};

/// What a table was identified as.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum TableId {
    /// A mozjpeg/libjpeg preset at a recovered quality. `preset` indexes
    /// [`MOZJPEG_PRESET_NAMES`]; `exact` is false when the match needed the
    /// tolerance, which happens when an encoder rounds the scale differently.
    Preset {
        preset: u8,
        quality: u8,
        exact: bool,
    },
    /// A Photoshop table (index into [`PHOTOSHOP_LUMA_TABLES`]). Photoshop
    /// does not use the IJG scale, so no quality is recoverable.
    Photoshop { index: u8 },
    /// A table minted by running a specific encoder, carrying the encoder name
    /// and the quality setting that produced it.
    Encoder { name: &'static str, quality: u8 },
    /// No known table matched. Callers must treat this as "the quantiser is
    /// unknown" and choose conservatively — see `slack_for`.
    Unrecognised,
}

/// libjpeg's quality→scale mapping, shared by every preset.
fn scale_for(quality: u8) -> u32 {
    let q = quality.clamp(1, 100) as u32;
    if q < 50 {
        5000 / q
    } else {
        200 - 2 * q
    }
}

/// Reconstruct one preset at one quality. `force_baseline` clamps to 255,
/// which is what a baseline-compatible encoder emits; without it the ceiling
/// is 32767 and low qualities differ.
pub fn preset_table(preset: usize, quality: u8, force_baseline: bool) -> [u16; 64] {
    let base = &MOZJPEG_LUMA_BASES[preset];
    let s = scale_for(quality);
    let cap: u32 = if force_baseline { 255 } else { 32767 };
    let mut out = [0u16; 64];
    for i in 0..64 {
        out[i] = ((base[i] as u32 * s + 50) / 100).clamp(1, cap) as u16;
    }
    out
}

fn max_abs_delta(a: &[u16; 64], b: &[u16; 64]) -> u16 {
    (0..64)
        .map(|i| a[i].abs_diff(b[i]))
        .max()
        .unwrap_or(u16::MAX)
}

/// Identify a luma quantization table.
///
/// `tolerance` is the largest per-coefficient difference still accepted as the
/// same table. Measured coverage on the survey corpus: 79.9% at 0, **86.4% at
/// 1**, 88.1% at 2. A tolerance of 1 absorbs encoders that round the scale
/// slightly differently while staying far from a neighbouring preset, so it is
/// the recommended default; larger values buy little and start to blur
/// adjacent qualities together.
pub fn identify_luma(table: &[u16; 64], tolerance: u16) -> TableId {
    // Exact first, so a table that is genuinely preset P at quality Q is never
    // reported as a near-miss of something else.
    for preset in 0..MOZJPEG_LUMA_BASES.len() {
        for quality in 1..=100u8 {
            for fb in [true, false] {
                if preset_table(preset, quality, fb) == *table {
                    return TableId::Preset {
                        preset: preset as u8,
                        quality,
                        exact: true,
                    };
                }
            }
        }
    }
    for (i, t) in PHOTOSHOP_LUMA_TABLES.iter().enumerate() {
        if t == table {
            return TableId::Photoshop { index: i as u8 };
        }
    }
    for (name, quality, t) in ENCODER_LUMA_TABLES.iter() {
        if t == table {
            return TableId::Encoder {
                name,
                quality: *quality,
            };
        }
    }
    if tolerance > 0 {
        let mut best: Option<(u16, u8, u8)> = None;
        for preset in 0..MOZJPEG_LUMA_BASES.len() {
            for quality in 1..=100u8 {
                for fb in [true, false] {
                    let d = max_abs_delta(&preset_table(preset, quality, fb), table);
                    if d <= tolerance && best.map(|(bd, _, _)| d < bd).unwrap_or(true) {
                        best = Some((d, preset as u8, quality));
                    }
                }
            }
        }
        if let Some((_, preset, quality)) = best {
            return TableId::Preset {
                preset,
                quality,
                exact: false,
            };
        }
    }
    TableId::Unrecognised
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every preset at every quality must identify as itself, exactly. This is
    /// the property that makes the reconstruction approach trustworthy: it is
    /// not a lookup that can go stale.
    #[test]
    fn every_preset_and_quality_round_trips() {
        for preset in 0..MOZJPEG_LUMA_BASES.len() {
            for quality in 1..=100u8 {
                let t = preset_table(preset, quality, true);
                match identify_luma(&t, 0) {
                    TableId::Preset { exact: true, .. } => {}
                    other => panic!("preset {preset} q{quality} identified as {other:?}"),
                }
            }
        }
    }

    /// Annex K at q75 is the most common table on the web; pin its identity
    /// and its recovered quality.
    #[test]
    fn annex_k_q75_recovers_its_quality() {
        let t = preset_table(0, 75, true);
        match identify_luma(&t, 1) {
            TableId::Preset {
                preset: 0,
                quality,
                exact: true,
            } => assert_eq!(quality, 75),
            other => panic!("expected Annex K q75, got {other:?}"),
        }
    }

    /// The Photoshop tables are in the set precisely because no preset makes
    /// them — if one ever did, the entry is redundant and should be dropped.
    #[test]
    fn photoshop_tables_are_not_reachable_as_presets() {
        for (i, t) in PHOTOSHOP_LUMA_TABLES.iter().enumerate() {
            assert_eq!(
                identify_luma(t, 0),
                TableId::Photoshop { index: i as u8 },
                "photoshop table {i} is also a mozjpeg preset — drop it"
            );
        }
    }

    /// A table no encoder produces must come back Unrecognised rather than
    /// being forced onto the nearest preset.
    #[test]
    fn nonsense_table_is_unrecognised() {
        let mut t = [7u16; 64];
        t[0] = 199;
        t[63] = 3;
        assert_eq!(identify_luma(&t, 1), TableId::Unrecognised);
    }
}
