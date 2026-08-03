//! jpegli-lineage quantiser parameters, for recovering the encoder's distance
//! from a table in closed form.
//!
//! zenjpeg (and cjpegli, sharing the lineage) builds each luma quantizer as
//!
//! ```text
//! q[i] = round(base[i] * GLOBAL_SCALE * freq_scale(d, i))
//! freq_scale(d, i) = d                                 if d < DIST_THRESHOLD
//!                  = max(0.5d, T^(1-e[i]) * d^e[i])    otherwise
//! ```
//!
//! The parameter is a continuous float, so the reachable table set cannot be
//! enumerated — sampling zenjpeg at 0.01 quality steps already yields 3,690
//! distinct tables against a ceiling near 11,400. These 128 floats invert the
//! whole space instead, which is why no jpegli tables are stored.
//!
//! Mirrored from zenjpeg (`foundation::consts::BASE_QUANT_MATRIX_YCBCR`,
//! `quant::FREQUENCY_EXPONENT`), which does not export them publicly.

/// Threshold where the frequency-dependent branch takes over.
pub const DIST_THRESHOLD: f32 = 1.5;
/// Global scale for the YCbCr path.
pub const GLOBAL_SCALE_YCBCR: f32 = 1.739_660_1;

/// Luma base quantization matrix (natural order).
pub static BASE_QUANT_LUMA: [f32; 64] = [
    1.239741, 1.722711, 2.921217, 2.812737, 3.33982, 3.463604, 3.840915, 3.86956, 1.722711,
    2.092889, 2.845676, 2.704507, 3.440767, 3.166232, 4.025209, 4.035324, 2.921217, 2.845676,
    2.95874, 3.386295, 3.619524, 3.904628, 3.757836, 4.237448, 2.812737, 2.704507, 3.386295,
    3.380059, 4.167987, 4.805511, 4.784259, 4.605934, 3.33982, 3.440767, 3.619524, 4.167987,
    4.579851, 4.923237, 5.574107, 5.485333, 3.463604, 3.166232, 3.904628, 4.805511, 4.923237,
    5.43936, 5.093896, 6.087225, 3.840915, 4.025209, 3.757836, 4.784259, 5.574107, 5.093896,
    5.438461, 5.403736, 3.86956, 4.035324, 4.237448, 4.605934, 5.485333, 6.087225, 5.403736,
    4.377871,
];

/// Per-coefficient exponent controlling the high-distance branch.
pub static FREQUENCY_EXPONENT: [f32; 64] = [
    1.0, 0.51, 0.67, 0.74, 1.0, 1.0, 1.0, 1.0, 0.51, 0.66, 0.69, 0.87, 1.0, 1.0, 1.0, 1.0, 0.67,
    0.69, 0.84, 0.83, 0.96, 1.0, 1.0, 1.0, 0.74, 0.87, 0.83, 1.0, 1.0, 0.91, 0.91, 1.0, 1.0, 1.0,
    0.96, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.91, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.91,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
];
