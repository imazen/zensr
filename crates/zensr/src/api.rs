//! The restoration API.
//!
//! Callers declare *intent* and what they *know about the file*; the library
//! owns policy. Nothing here exposes a research constant, because a caller has
//! no way to reason about one — they cannot know whether a projection slack of
//! 0.15 is right, but they can know whether they just encoded the file.
//!
//! Current entry points and result semantics: `docs/API.md`.

use crate::calibration::*;
use crate::{Projection, RestoreConfig, RestoreError};

/// What the caller is optimising for.
///
/// The enum has no `Default` implementation. Requests start with Fidelity and
/// the routing policy selected for this crate; callers can override the intent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Intent {
    /// Maximise measured closeness to the pre-compression original.
    ///
    /// Default requests use decoded-feature routing over native-resolution
    /// analysis windows. Explicit overrides retain coefficient/quality routing.
    /// Individual files can still come out worse; this is not a guarantee.
    Fidelity,
    /// Prefer leaving a file alone when the expected gain is small.
    ///
    /// Uses the existing coefficient/quality policy with an earlier identity
    /// gate. This reduces coverage compared with that policy's Fidelity mode;
    /// it does not promise lower harm than the decoded-feature default.
    Conservative,
}

/// What the caller knows about where these bytes came from.
///
/// This cannot be measured from the file. Generation detection was
/// implemented and falsified: there is no operating point where a
/// conservative answer exists (zero false-gen1 implies zero gen1 recall), and
/// same-encoder equal-quality recompression is essentially invisible at 9.4%
/// recall. So it is an input, and absence of evidence is never read as
/// "freshly encoded".
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Provenance {
    /// Third-party file of unknown history. The safe default.
    Unknown,
    /// This process encoded it, once, from an uncompressed source.
    FreshEncode,
    // `Generations(u8)` was here and has been REMOVED, not deferred.
    //
    // Two separate reasons, and the second is the disqualifying one:
    //
    // 1. Generation count is not recoverable from a file. Detection was
    //    implemented and falsified — no operating point exists where a
    //    conservative answer is possible, and resizing (which CDNs do
    //    constantly) destroys the double-quantisation periodicity it relies
    //    on. So only a caller could supply it, and few can.
    // 2. Nothing validated existed to DO with it. The measurement is "each
    //    generation adds roughly 1-2.5 quantizer steps of excess" — an
    //    ABSOLUTE figure, i.e. `slack_abs`. A first implementation here wired
    //    it into `slack_q`, which is a FRACTION of the quantizer, with an
    //    invented per-generation coefficient. Wrong units and a fabricated
    //    constant.
    //
    // Treating it as `Unknown` would have been safe but decorative: identical
    // behaviour under a different name. Restore the variant when the
    // n -> slack_abs mapping is measured end to end (ROADMAP 1.9).
}

/// Select a caller-loaded model tier. The library does not load weights itself.
/// Current routing calibration targets canon4; other model weights need their
/// own quality measurements. Runtime cost depends on the selected model and host.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Budget {
    /// Compact tier; canon4 uses 86,406 bytes of f16 weights.
    Realtime,
    /// The caller's quality-oriented model.
    Quality,
}

/// Why a file was returned unmodified.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum SkipReason {
    /// The input is near-pristine for its chroma subsampling, where the model
    /// is measured to lose frequently. Thresholds depend on the corrected
    /// probe family, quality scale, and subsampling.
    NearPristine,
    /// Consistency was required but cannot be certified for this file's
    /// chroma geometry.
    ConsistencyUnavailable(NotCertified),
    /// The default policy declined restoration, [`Routing::Auto`] estimated
    /// benefit below the caller's threshold, or [`Routing::Never`] was set.
    /// Carries the policy score: risk-adjusted gain for the decoded default,
    /// or the legacy estimated gain for an explicit Auto override.
    NotWorthIt { estimated_gain: f32 },
    /// The colour model is not one restoration understands — 4-component
    /// (CMYK/YCCK) files are decoded and returned untouched.
    UnsupportedColorModel,
}

/// Why the quantisation box could not be certified.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum NotCertified {
    /// 4:2:2 and 4:4:0 chroma are not back-projected today.
    ChromaGeometry,
    /// The caller opted out.
    CallerDisabled,
}

/// Whether coefficient projection was applied to the output.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum Consistency {
    /// Projection was applied to all three YCbCr components using the
    /// family-calibrated slack. This is not a bit-exact re-encoding guarantee:
    /// slack expands the quantisation intervals, and RGB conversion and
    /// rounding can move coefficients after projection.
    ///
    /// This records application of the projection, not a final RGB8 bound
    /// check or a guarantee of improved perceptual quality.
    Certified,
    /// Projection was not applied; `why` is specific.
    NotEnforced(NotCertified),
}

/// Chroma subsampling, as a routing input rather than trivia: it decides
/// whether restoration helps at all at a given quality.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Chroma {
    /// 4:4:4 — full chroma resolution. Least damaged at a given quality, so
    /// the near-pristine threshold is *lower* here.
    Full,
    /// 4:2:0 — chroma halved in both axes. The common web case.
    HalfBoth,
    /// 4:2:2, 4:4:0, or anything else. Not back-projected today.
    Other,
}

/// Why the library did what it did. Every field answers "why did it do that",
/// which is what was missing during two weeks of measurement.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct Report {
    /// Encoder family as the probe names it, from markers and structure.
    ///
    /// Note this describes provenance, not the quantiser: measured over 17,739
    /// files the probe reports `ImageMagick` for 9,626 of which only 180 use
    /// the ImageMagick table. Use `quality` for quantiser behaviour.
    pub encoder_family: String,
    /// Estimated quality on that family's own scale.
    pub quality: f32,
    /// Which scale `quality` is on — IJG quality, or a Butteraugli distance.
    pub quality_scale: String,
    /// Chroma subsampling; see [`Chroma`].
    pub chroma: Chroma,
    /// Whether the coefficient-domain deblock ran.
    pub deblocked: bool,
    /// Fraction of projected coefficients that were clamped, per component.
    pub clamped_fraction: Vec<f32>,
}

/// The result of a restoration request.
///
/// `Unchanged` is a first-class outcome, not an error and not a flag: on
/// high-quality traffic it is the common path, and a CDN can skip re-encoding
/// entirely when it sees one.
#[derive(Debug)]
pub enum Outcome {
    /// Policy determined restoration would not help, or could not be applied
    /// safely. `pixels` is the plain decode.
    Unchanged {
        pixels: Pixels,
        why: SkipReason,
        report: Report,
    },
    /// Restoration applied.
    Restored {
        pixels: Pixels,
        consistency: Consistency,
        report: Report,
    },
}

impl Outcome {
    /// A lightweight outcome record without retaining a second image buffer.
    pub fn summary(&self) -> Summary {
        Summary {
            applied: matches!(self, Self::Restored { .. }),
            reason: match self {
                Self::Restored { .. } => "Restoration and coefficient projection applied".into(),
                Self::Unchanged { why, .. } => format!("Restoration bypassed: {why:?}"),
            },
            report: self.report().clone(),
        }
    }
    /// Consume the result without cloning its pixel planes.
    pub fn into_pixels(self) -> Pixels {
        match self {
            Self::Restored { pixels, .. } | Self::Unchanged { pixels, .. } => pixels,
        }
    }
    /// The pixels, whichever arm this is.
    #[must_use]
    pub fn pixels(&self) -> &Pixels {
        match self {
            Outcome::Unchanged { pixels, .. } | Outcome::Restored { pixels, .. } => pixels,
        }
    }

    /// The report, whichever arm this is.
    #[must_use]
    pub fn report(&self) -> &Report {
        match self {
            Outcome::Unchanged { report, .. } | Outcome::Restored { report, .. } => report,
        }
    }
}

/// Codec-visible restoration status, separate from decoded sample storage.
#[derive(Clone, Debug)]
pub struct Summary {
    pub applied: bool,
    pub reason: String,
    pub report: Report,
}

fn write_planar_to_rgb8_strided(
    planes: &[f32],
    width: usize,
    height: usize,
    dst: &mut [u8],
    stride_bytes: usize,
) -> Result<(), RestoreError> {
    let row_bytes = width
        .checked_mul(3)
        .ok_or(RestoreError::UnsupportedPixels("RGB8 row width overflow"))?;
    if stride_bytes < row_bytes {
        return Err(RestoreError::UnsupportedPixels(
            "RGB8 destination stride smaller than width * 3",
        ));
    }
    let needed = if height == 0 {
        0
    } else {
        (height - 1)
            .checked_mul(stride_bytes)
            .and_then(|n| n.checked_add(row_bytes))
            .ok_or(RestoreError::UnsupportedPixels(
                "RGB8 destination size overflow",
            ))?
    };
    if dst.len() < needed {
        return Err(RestoreError::UnsupportedPixels(
            "RGB8 destination buffer too small for dimensions and stride",
        ));
    }
    let plane = width * height;
    let (r_plane, rest) = planes.split_at(plane);
    let (g_plane, b_plane) = rest.split_at(plane);
    for y in 0..height {
        let row = &mut dst[y * stride_bytes..y * stride_bytes + row_bytes];
        let p_off = y * width;
        for x in 0..width {
            row[x * 3] = (r_plane[p_off + x] * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
            row[x * 3 + 1] = (g_plane[p_off + x] * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
            row[x * 3 + 2] = (b_plane[p_off + x] * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
        }
    }
    Ok(())
}

fn write_planar_to_rgb_f32_strided(
    planes: &[f32],
    width: usize,
    height: usize,
    dst: &mut [f32],
    stride_floats: usize,
) -> Result<(), RestoreError> {
    let row_floats = width.checked_mul(3).ok_or(RestoreError::UnsupportedPixels(
        "RGB f32 row width overflow",
    ))?;
    if stride_floats < row_floats {
        return Err(RestoreError::UnsupportedPixels(
            "RGB f32 destination stride smaller than width * 3",
        ));
    }
    let needed = if height == 0 {
        0
    } else {
        (height - 1)
            .checked_mul(stride_floats)
            .and_then(|n| n.checked_add(row_floats))
            .ok_or(RestoreError::UnsupportedPixels(
                "RGB f32 destination size overflow",
            ))?
    };
    if dst.len() < needed {
        return Err(RestoreError::UnsupportedPixels(
            "RGB f32 destination buffer too small for dimensions and stride",
        ));
    }
    let plane = width * height;
    let (r_plane, rest) = planes.split_at(plane);
    let (g_plane, b_plane) = rest.split_at(plane);
    for y in 0..height {
        let row = &mut dst[y * stride_floats..y * stride_floats + row_floats];
        let p_off = y * width;
        for x in 0..width {
            row[x * 3] = r_plane[p_off + x];
            row[x * 3 + 1] = g_plane[p_off + x];
            row[x * 3 + 2] = b_plane[p_off + x];
        }
    }
    Ok(())
}

#[cfg(feature = "zenpixels")]
fn write_planar_to_pixel_slice<P>(
    planes: &[f32],
    width: usize,
    height: usize,
    dst: &mut zenpixels::PixelSliceMut<'_, P>,
) -> Result<(), RestoreError> {
    if dst.width() as usize != width || dst.rows() as usize != height {
        return Err(RestoreError::UnsupportedPixels(
            "destination PixelSliceMut dimensions do not match restored image",
        ));
    }
    let desc = dst.descriptor();
    if desc.layout() != zenpixels::ChannelLayout::Rgb {
        return Err(RestoreError::UnsupportedPixels(
            "destination PixelSliceMut must use 3-channel Rgb layout",
        ));
    }
    let plane = width * height;
    let (r_plane, rest) = planes.split_at(plane);
    let (g_plane, b_plane) = rest.split_at(plane);
    match desc.channel_type() {
        zenpixels::ChannelType::U8 => {
            for y in 0..height {
                let row = dst.row_mut(y as u32);
                let p_off = y * width;
                for x in 0..width {
                    row[x * 3] = (r_plane[p_off + x] * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
                    row[x * 3 + 1] = (g_plane[p_off + x] * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
                    row[x * 3 + 2] = (b_plane[p_off + x] * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
                }
            }
            Ok(())
        }
        zenpixels::ChannelType::F32 => {
            for y in 0..height {
                let row = dst.row_mut(y as u32);
                let p_off = y * width;
                for (x, px_bytes) in row.as_chunks_mut::<12>().0.iter_mut().enumerate() {
                    px_bytes[0..4].copy_from_slice(&r_plane[p_off + x].to_ne_bytes());
                    px_bytes[4..8].copy_from_slice(&g_plane[p_off + x].to_ne_bytes());
                    px_bytes[8..12].copy_from_slice(&b_plane[p_off + x].to_ne_bytes());
                }
            }
            Ok(())
        }
        _ => Err(RestoreError::UnsupportedPixels(
            "destination PixelSliceMut channel type must be U8 or F32",
        )),
    }
}

/// Borrowed planar f32 pixels (`[3, height, width]` in `0.0..=1.0`) backed by a
/// [`RepairWorkspace`] or [`Pixels`].
///
/// Samples remain in the JPEG's coding RGB space prior to ICC profile
/// conversion; callers using the `zenpixels` helpers should attach or preserve
/// color context on the codec side.
#[derive(Clone, Copy, Debug)]
pub struct PixelsRef<'a> {
    planes: &'a [f32],
    width: usize,
    height: usize,
}

impl<'a> PixelsRef<'a> {
    #[must_use]
    pub fn width(&self) -> usize {
        self.width
    }

    #[must_use]
    pub fn height(&self) -> usize {
        self.height
    }

    /// Planar f32, three planes of `width * height`, values in `0.0..=1.0`.
    #[must_use]
    pub fn planes(&self) -> &'a [f32] {
        self.planes
    }

    /// Allocate tightly packed interleaved `f32` RGB (`width * height * 3`).
    #[must_use]
    pub fn to_rgb_f32(&self) -> Vec<f32> {
        let mut out = vec![0.0f32; self.width * self.height * 3];
        write_planar_to_rgb_f32_strided(
            self.planes,
            self.width,
            self.height,
            &mut out,
            self.width * 3,
        )
        .expect("tightly packed RGB f32 buffer matches dimensions");
        out
    }

    /// Allocate tightly packed interleaved 8-bit RGB (`width * height * 3`).
    #[must_use]
    pub fn to_rgb8(&self) -> Vec<u8> {
        let mut out = vec![0u8; self.width * self.height * 3];
        write_planar_to_rgb8_strided(
            self.planes,
            self.width,
            self.height,
            &mut out,
            self.width * 3,
        )
        .expect("tightly packed RGB8 buffer matches dimensions");
        out
    }

    /// Write interleaved 8-bit RGB into caller-owned strided storage with zero
    /// heap allocations. Row padding bytes (`width * 3 .. stride_bytes`) are
    /// left untouched.
    pub fn write_rgb8_strided(
        &self,
        dst: &mut [u8],
        stride_bytes: usize,
    ) -> Result<(), RestoreError> {
        write_planar_to_rgb8_strided(self.planes, self.width, self.height, dst, stride_bytes)
    }

    /// Write interleaved `f32` RGB into caller-owned strided storage (`stride_floats >= width * 3`)
    /// with zero heap allocations. Row padding elements (`width * 3 .. stride_floats`)
    /// are left untouched.
    pub fn write_rgb_f32_strided(
        &self,
        dst: &mut [f32],
        stride_floats: usize,
    ) -> Result<(), RestoreError> {
        write_planar_to_rgb_f32_strided(self.planes, self.width, self.height, dst, stride_floats)
    }

    /// Write interleaved RGB (`U8` or `F32`) directly into a caller-owned
    /// [`zenpixels::PixelSliceMut`] (typed or type-erased) with zero heap
    /// allocations, preserving any row-stride padding.
    #[cfg(feature = "zenpixels")]
    pub fn write_pixel_slice<P>(
        &self,
        dst: &mut zenpixels::PixelSliceMut<'_, P>,
    ) -> Result<(), RestoreError> {
        write_planar_to_pixel_slice(self.planes, self.width, self.height, dst)
    }

    /// Allocate an owned [`zenpixels::PixelBuffer<rgb::Rgb<u8>>`] tagged with
    /// [`zenpixels::PixelDescriptor::RGB8`] (`TransferFunction::Unknown`, coding RGB).
    #[cfg(feature = "zenpixels")]
    pub fn to_pixel_buffer_rgb8(
        &self,
    ) -> Result<zenpixels::PixelBuffer<rgb::Rgb<u8>>, zenpixels::BufferError> {
        let mut buf = zenpixels::PixelBuffer::<rgb::Rgb<u8>>::try_new_typed(
            self.width as u32,
            self.height as u32,
        )
        .map_err(|e| e.decompose().0)?;
        {
            let mut slice = buf.as_slice_mut();
            write_planar_to_pixel_slice(self.planes, self.width, self.height, &mut slice)
                .expect("freshly allocated Rgb<u8> PixelBuffer matches dimensions");
        }
        Ok(buf)
    }

    /// Allocate an owned [`zenpixels::PixelBuffer<rgb::Rgb<f32>>`] tagged with
    /// [`zenpixels::PixelDescriptor::RGBF32`] (`TransferFunction::Unknown`, coding RGB).
    #[cfg(feature = "zenpixels")]
    pub fn to_pixel_buffer_rgb_f32(
        &self,
    ) -> Result<zenpixels::PixelBuffer<rgb::Rgb<f32>>, zenpixels::BufferError> {
        let mut buf = zenpixels::PixelBuffer::<rgb::Rgb<f32>>::try_new_typed(
            self.width as u32,
            self.height as u32,
        )
        .map_err(|e| e.decompose().0)?;
        {
            let mut slice = buf.as_slice_mut();
            write_planar_to_pixel_slice(self.planes, self.width, self.height, &mut slice)
                .expect("freshly allocated Rgb<f32> PixelBuffer matches dimensions");
        }
        Ok(buf)
    }
}

/// Decoded pixels, planar f32 in 0..=1, three components.
#[derive(Debug)]
pub struct Pixels {
    planes: Vec<f32>,
    width: usize,
    height: usize,
}

impl Pixels {
    /// Borrow these pixels as a [`PixelsRef`].
    #[must_use]
    pub fn as_pixels_ref(&self) -> PixelsRef<'_> {
        PixelsRef {
            planes: &self.planes,
            width: self.width,
            height: self.height,
        }
    }

    #[must_use]
    pub fn width(&self) -> usize {
        self.width
    }
    #[must_use]
    pub fn height(&self) -> usize {
        self.height
    }
    /// Planar f32, three planes of `width * height`, values in 0..=1.
    #[must_use]
    pub fn planes(&self) -> &[f32] {
        &self.planes
    }
    /// Consume unquantized planes into interleaved f32 RGB.
    pub fn into_rgb_f32(self) -> Vec<f32> {
        self.as_pixels_ref().to_rgb_f32()
    }
    /// Allocate tightly packed interleaved `f32` RGB (`width * height * 3`).
    #[must_use]
    pub fn to_rgb_f32(&self) -> Vec<f32> {
        self.as_pixels_ref().to_rgb_f32()
    }
    /// Interleaved 8-bit RGB.
    #[must_use]
    pub fn to_rgb8(&self) -> Vec<u8> {
        self.as_pixels_ref().to_rgb8()
    }
    /// Write interleaved 8-bit RGB into caller-owned strided storage with zero
    /// heap allocations. Row padding bytes (`width * 3 .. stride_bytes`) are
    /// left untouched.
    pub fn write_rgb8_strided(
        &self,
        dst: &mut [u8],
        stride_bytes: usize,
    ) -> Result<(), RestoreError> {
        self.as_pixels_ref().write_rgb8_strided(dst, stride_bytes)
    }
    /// Write interleaved `f32` RGB into caller-owned strided storage (`stride_floats >= width * 3`)
    /// with zero heap allocations.
    pub fn write_rgb_f32_strided(
        &self,
        dst: &mut [f32],
        stride_floats: usize,
    ) -> Result<(), RestoreError> {
        self.as_pixels_ref()
            .write_rgb_f32_strided(dst, stride_floats)
    }
    /// Write interleaved RGB (`U8` or `F32`) directly into a caller-owned
    /// [`zenpixels::PixelSliceMut`] (typed or type-erased) with zero heap
    /// allocations, preserving any row-stride padding.
    #[cfg(feature = "zenpixels")]
    pub fn write_pixel_slice<P>(
        &self,
        dst: &mut zenpixels::PixelSliceMut<'_, P>,
    ) -> Result<(), RestoreError> {
        self.as_pixels_ref().write_pixel_slice(dst)
    }
    /// Allocate an owned [`zenpixels::PixelBuffer<rgb::Rgb<u8>>`] tagged with
    /// [`zenpixels::PixelDescriptor::RGB8`] (`TransferFunction::Unknown`, coding RGB).
    #[cfg(feature = "zenpixels")]
    pub fn to_pixel_buffer_rgb8(
        &self,
    ) -> Result<zenpixels::PixelBuffer<rgb::Rgb<u8>>, zenpixels::BufferError> {
        self.as_pixels_ref().to_pixel_buffer_rgb8()
    }
    /// Allocate an owned [`zenpixels::PixelBuffer<rgb::Rgb<f32>>`] tagged with
    /// [`zenpixels::PixelDescriptor::RGBF32`] (`TransferFunction::Unknown`, coding RGB).
    #[cfg(feature = "zenpixels")]
    pub fn to_pixel_buffer_rgb_f32(
        &self,
    ) -> Result<zenpixels::PixelBuffer<rgb::Rgb<f32>>, zenpixels::BufferError> {
        self.as_pixels_ref().to_pixel_buffer_rgb_f32()
    }
}

/// Estimated projected restoration gain for the canon4 realtime model, in
/// SSIM2 points. Uses the record-corrected header probe without model inference.
///
/// Curves are monotone fits to calibration-image medians, not guarantees for
/// individual files. Auto routing also measures the luma zero-AC fraction.
/// See `benchmarks/ship_rt24i_canon4_2026-11-04.md` for held-out results and
/// the quality/work/harm tradeoff. Other models require separate calibration.
#[must_use]
pub fn estimate_gain(probe: &crate::source::JpegInfo) -> f32 {
    let _ = crate::record::is_recorded(probe);
    estimate_gain_for(probe, None)
}

/// Coefficient-domain content class used by the calibrated routing curves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Content {
    Graphic,
    Photographic,
}

fn classify_coefficients_view(coeffs: &crate::source::CoefficientsView<'_>) -> Option<Content> {
    let luma = coeffs.first()?;
    let nblocks = luma.coeffs.len() / 64;
    if nblocks == 0 {
        return None;
    }
    let zero_ac = luma.coeffs[..nblocks * 64]
        .as_chunks::<64>()
        .0
        .iter()
        .filter(|b| b[1..].iter().all(|&c| c == 0))
        .count();
    Some(
        if zero_ac as f32 / nblocks as f32 >= GRAPHIC_ZERO_AC_THRESHOLD {
            Content::Graphic
        } else {
            Content::Photographic
        },
    )
}

/// [`estimate_gain`] with an optional content class. `None` reproduces the
/// quality-only estimate exactly.
pub(crate) fn estimate_gain_for(probe: &crate::source::JpegInfo, content: Option<Content>) -> f32 {
    let full = crate::chroma_full_of(probe) == Some(true);
    let scale = probe.quality.scale.as_str();
    if scale == "ButteraugliDistance" {
        // The AQ term is zero without a verified record (record.rs).
        return gain_at_distance(full, probe.quality.value, content)
            + crate::record::aq_off_gain_delta(probe);
    }
    if scale != "IjgQuality" && scale != "MozjpegQuality" {
        // An uncalibrated scale is not an IJG quality dial.
        return 0.0;
    }
    let q = probe.quality.value;
    match content {
        None => gain_at(full, q),
        Some(Content::Graphic) => lerp(if full { GRAPHIC444 } else { GRAPHIC420 }, q),
        Some(Content::Photographic) => lerp(if full { PHOTO444 } else { PHOTO420 }, q),
    }
}

/// Gain curve for distance-quantised encoders, separated from probing so it can
/// be tested at exact distances.
fn gain_at_distance(full: bool, d: f32, content: Option<Content>) -> f32 {
    let pts = match (content, full) {
        (Some(Content::Graphic), false) => GRAPHIC_DIST420,
        (Some(Content::Graphic), true) => GRAPHIC_DIST444,
        (Some(Content::Photographic), false) => PHOTO_DIST420,
        (Some(Content::Photographic), true) => PHOTO_DIST444,
        (None, false) => DIST420,
        (None, true) => DIST444,
    };
    lerp(pts, d)
}

/// Linear interpolation over an ascending table, clamped at both ends.
pub(crate) fn lerp(pts: &[(f32, f32)], x: f32) -> f32 {
    if x <= pts[0].0 {
        return pts[0].1;
    }
    for w in pts.windows(2) {
        let ((x0, y0), (x1, y1)) = (w[0], w[1]);
        if x <= x1 {
            let t = if x1 > x0 { (x - x0) / (x1 - x0) } else { 0.0 };
            return y0 + t * (y1 - y0);
        }
    }
    pts[pts.len() - 1].1
}

/// The gain curve itself, separated from probing so it can be tested at exact
/// qualities.
///
/// Going through a real encode to reach it does not test this: the probe
/// recovers a quality a point or two off what the encoder was asked for, and
/// that slack is indistinguishable from a mistyped anchor. Both are worth
/// testing — this is the half that says what the calibration claims.
fn gain_at(full: bool, q: f32) -> f32 {
    lerp(if full { G444 } else { G420 }, q)
}

/// When to spend cycles on restoration.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum Routing {
    /// Restore whenever the near-pristine gate allows it.
    Always,
    /// Never restore; decode and return. Useful for A/B and for kill switches.
    Never,
    /// Restore when the content-aware estimate reaches `min_gain` SSIM2 and
    /// the near-pristine gate permits it. Larger thresholds save work and
    /// suppress some regressions, but also discard real gains.
    Auto { min_gain: f32 },
}

impl Default for Routing {
    /// Explicit legacy [`Routing::Auto`] at 0.25 SSIM2. Omitting a request's
    /// routing override selects the default request policy instead.
    fn default() -> Self {
        Routing::Auto { min_gain: 0.25 }
    }
}

/// Per-image restoration choices shared by codec and direct engine callers.
#[derive(Clone, Copy, Debug)]
pub struct Options {
    intent: Intent,
    provenance: Provenance,
    budget: Budget,
    require_consistency: bool,
    routing: Option<Routing>,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            intent: Intent::Fidelity,
            provenance: Provenance::Unknown,
            budget: Budget::Realtime,
            require_consistency: false,
            routing: None,
        }
    }
}
impl Options {
    /// Set the requested quality/coverage tradeoff.
    pub fn intent(mut self, value: Intent) -> Self {
        self.intent = value;
        self
    }
    /// Declare what the caller knows about the encoded input.
    pub fn provenance(mut self, value: Provenance) -> Self {
        self.provenance = value;
        self
    }
    /// Select the caller-loaded model tier.
    pub fn budget(mut self, value: Budget) -> Self {
        self.budget = value;
        self
    }
    /// Require projection of every supported component.
    pub fn require_consistency(mut self, value: bool) -> Self {
        self.require_consistency = value;
        self
    }
    /// Override learned routing explicitly.
    pub fn routing(mut self, value: Routing) -> Self {
        self.routing = Some(value);
        self
    }
}

/// A reusable restorer. Build once, use for many images.
///
/// Holds one model per [`Budget`] tier the caller chooses to load. Requesting
/// a tier that was not loaded is an error rather than a silent downgrade —
/// quietly serving 84 KB weights to a caller who asked for the 1.16 MB tier
/// would misreport quality with no way to notice.
pub struct Restorer {
    realtime: Option<zensr_micro::adopted::AdoptedModel>,
    quality: Option<zensr_micro::adopted::AdoptedModel>,
}

impl Restorer {
    /// An empty restorer; add tiers with [`Restorer::with_tier`].
    #[must_use]
    pub fn new() -> Self {
        Self {
            realtime: None,
            quality: None,
        }
    }

    /// Load weights for a tier. Which tier a set of weights belongs to is the
    /// caller's to track; it is not derivable from the bytes.
    #[must_use]
    pub fn with_tier(mut self, tier: Budget, model: zensr_micro::adopted::AdoptedModel) -> Self {
        match tier {
            Budget::Realtime => self.realtime = Some(model),
            Budget::Quality => self.quality = Some(model),
        }
        self
    }

    fn tier(&self, b: Budget) -> Option<&zensr_micro::adopted::AdoptedModel> {
        match b {
            Budget::Realtime => self.realtime.as_ref(),
            Budget::Quality => self.quality.as_ref(),
        }
    }

    /// Begin a restoration request.
    #[must_use]
    pub fn restore<'a>(&'a self, source: &'a mut dyn crate::source::Source) -> Request<'a> {
        Request {
            restorer: self,
            source,
            intent: Intent::Fidelity,
            provenance: Provenance::Unknown,
            budget: Budget::Realtime,
            routing: None,
            require_consistency: false,
            threads: 0,
            workspace: None,
        }
    }
}

/// Reusable scratch buffers for zero-allocation steady-state restoration.
///
/// Allocate one `RepairWorkspace` per worker thread (or pipeline instance) and
/// pass it across images via [`Request::run_into_workspace`] (zero warm heap
/// allocations when paired with caller-owned strided output buffers) or
/// [`Request::workspace`] (reuses internal scratch while returning an owned
/// [`Outcome`]).
#[derive(Debug)]
pub struct RepairWorkspace {
    pub(crate) model_ws: zensr_micro::adopted::ModelWorkspace,
    pub(crate) guard_scratch: zensr_micro::guards::GuardScratch,
    pub(crate) chroma_scratch: zensr_micro::consist::ChromaScratch,
    #[cfg(feature = "chooser")]
    pub(crate) router_scratch: crate::decoded_router::DecodedRouterScratch,
    pub(crate) buf_a: Vec<f32>,
    pub(crate) buf_b: Vec<f32>,
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) restore_report: crate::RestoreReport,
    pub(crate) api_report: Report,
}

impl RepairWorkspace {
    /// Create an empty workspace whose buffers grow lazily on first use and are
    /// reused across subsequent frames without reallocation.
    #[must_use]
    pub fn new() -> Self {
        Self {
            model_ws: zensr_micro::adopted::ModelWorkspace::new(),
            guard_scratch: zensr_micro::guards::GuardScratch::new(),
            chroma_scratch: zensr_micro::consist::ChromaScratch::new(),
            #[cfg(feature = "chooser")]
            router_scratch: crate::decoded_router::DecodedRouterScratch::default(),
            buf_a: Vec::new(),
            buf_b: Vec::new(),
            width: 0,
            height: 0,
            restore_report: crate::RestoreReport::default(),
            api_report: Report {
                encoder_family: String::new(),
                quality: 0.0,
                quality_scale: String::new(),
                chroma: Chroma::Other,
                deblocked: false,
                clamped_fraction: Vec::new(),
            },
        }
    }

    /// Create a pre-sized workspace for frames up to `max_width × max_height`
    /// and a specific [`AdoptedModel`](zensr_micro::adopted::AdoptedModel) /
    /// tile geometry (`tile == 0` uses 192, the largest automatic tile size).
    #[must_use]
    pub fn with_capacity(
        max_width: usize,
        max_height: usize,
        model: &zensr_micro::adopted::AdoptedModel,
        tile: usize,
    ) -> Self {
        let max_tile = if tile == 0 { 192 } else { tile };
        let tile_h = max_height.min(max_tile).max(32);
        let tile_w = max_width.min(max_tile).max(32);
        let plane = max_width.saturating_mul(max_height);
        let total = plane.saturating_mul(3);
        let mut ws = Self::new();
        ws.model_ws =
            zensr_micro::adopted::ModelWorkspace::with_tile_capacity(model, tile_h, tile_w);
        ws.guard_scratch = zensr_micro::guards::GuardScratch::with_capacity(max_height, max_width);
        ws.chroma_scratch = zensr_micro::consist::ChromaScratch::with_capacity(
            max_width.div_ceil(16) * 8,
            max_height.div_ceil(16) * 8,
        );
        ws.buf_a.reserve(total);
        ws.buf_b.reserve(total);
        ws.restore_report.encoder_family.reserve(64);
        ws.restore_report.quality_scale.reserve(32);
        ws.restore_report.projection.reserve(3);
        ws.api_report.encoder_family.reserve(64);
        ws.api_report.quality_scale.reserve(32);
        ws.api_report.clamped_fraction.reserve(3);
        ws
    }
}

impl Default for RepairWorkspace {
    fn default() -> Self {
        Self::new()
    }
}

/// Borrowed outcome of [`Request::run_into_workspace`], referencing pixels and
/// provenance metadata stored inside the caller's [`RepairWorkspace`].
#[derive(Debug)]
pub enum WorkspaceOutcome<'a> {
    /// Policy determined restoration would not help, or could not be applied
    /// safely. `pixels` is the plain decode.
    Unchanged {
        pixels: PixelsRef<'a>,
        why: SkipReason,
        report: &'a Report,
    },
    /// Restoration applied.
    Restored {
        pixels: PixelsRef<'a>,
        consistency: Consistency,
        report: &'a Report,
    },
}

impl<'a> WorkspaceOutcome<'a> {
    /// True when restoration ran (`WorkspaceOutcome::Restored`).
    #[must_use]
    pub fn applied(&self) -> bool {
        matches!(self, Self::Restored { .. })
    }

    /// Borrow the decoded or restored planar pixels.
    #[must_use]
    pub fn pixels(&self) -> PixelsRef<'a> {
        match *self {
            Self::Unchanged { pixels, .. } | Self::Restored { pixels, .. } => pixels,
        }
    }

    /// Borrow the provenance and policy report.
    #[must_use]
    pub fn report(&self) -> &'a Report {
        match self {
            Self::Unchanged { report, .. } | Self::Restored { report, .. } => report,
        }
    }

    /// Allocate an owned [`Summary`] snapshot.
    #[must_use]
    pub fn summary(&self) -> Summary {
        Summary {
            applied: self.applied(),
            reason: match self {
                Self::Restored { .. } => "Restoration and coefficient projection applied".into(),
                Self::Unchanged { why, .. } => format!("Restoration bypassed: {why:?}"),
            },
            report: self.report().clone(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum OutcomeKind {
    Unchanged(SkipReason),
    Restored(Consistency),
}

/// A pending restoration request.
pub struct Request<'a> {
    restorer: &'a Restorer,
    source: &'a mut dyn crate::source::Source,
    intent: Intent,
    provenance: Provenance,
    budget: Budget,
    routing: Option<Routing>,
    require_consistency: bool,
    threads: usize,
    workspace: Option<&'a mut RepairWorkspace>,
}

impl Default for Restorer {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> Request<'a> {
    /// Apply the same per-image choices used by codec integrations.
    pub fn options(mut self, options: Options) -> Self {
        self.intent = options.intent;
        self.provenance = options.provenance;
        self.budget = options.budget;
        self.require_consistency = options.require_consistency;
        self.routing = options.routing;
        self
    }

    /// What to optimise for. Defaults to [`Intent::Fidelity`].
    #[must_use]
    pub fn intent(mut self, intent: Intent) -> Self {
        self.intent = intent;
        self
    }

    /// What is known about the file's history. Defaults to
    /// [`Provenance::Unknown`], which is the safe reading.
    #[must_use]
    pub fn provenance(mut self, p: Provenance) -> Self {
        self.provenance = p;
        self
    }

    /// Compute budget. Defaults to [`Budget::Realtime`].
    #[must_use]
    pub fn budget(mut self, b: Budget) -> Self {
        self.budget = b;
        self
    }

    /// Return [`Outcome::Unchanged`] if coefficient projection cannot be applied
    /// to all three YCbCr components. This requires projection availability;
    /// it does not verify the bounds again after RGB conversion and rounding.
    /// See [`Consistency::Certified`] for the scope of that report.
    #[must_use]
    pub fn require_consistency(mut self, on: bool) -> Self {
        self.require_consistency = on;
        self
    }

    /// Override the default policy. With default features, Fidelity requests
    /// use the frozen decoded-feature router on native analysis windows.
    /// Explicit overrides retain their existing gain-threshold and gate rules.
    /// Conservative intent, unsupported analysis inputs and builds without
    /// `chooser` use calibrated Auto at 0.25. See the reserved routing report.
    #[must_use]
    pub fn routing(mut self, r: Routing) -> Self {
        self.routing = Some(r);
        self
    }

    /// Worker threads; 0 means choose automatically.
    #[must_use]
    pub fn threads(mut self, n: usize) -> Self {
        self.threads = n;
        self
    }

    /// Reuse `ws` scratch buffers when executing [`Request::run`].
    ///
    /// For zero-allocation steady-state execution that also avoids allocating
    /// an owned [`Outcome`] per frame, call [`Request::run_into_workspace`].
    #[must_use]
    pub fn workspace(mut self, ws: &'a mut RepairWorkspace) -> Self {
        self.workspace = Some(ws);
        self
    }

    fn execute_into_workspace(
        restorer: &Restorer,
        prepared: &mut dyn crate::source::Source,
        options: Options,
        threads: usize,
        ws: &mut RepairWorkspace,
    ) -> Result<OutcomeKind, RestoreError> {
        let Options {
            intent,
            provenance,
            budget,
            require_consistency,
            routing: routing_override,
        } = options;
        let model = restorer.tier(budget).ok_or(RestoreError::TierNotLoaded)?;

        // Routing decides before restoration inference. The learned default
        // also pays for plain decoded-pixel features; explicit overrides keep
        // their existing coefficient/quality routing behavior.
        let (recorded, geometry_unavailable) = {
            let probe = prepared.info();
            (
                crate::record::is_recorded(probe),
                crate::chroma_full_of(probe).is_none(),
            )
        };
        let routing = routing_override.unwrap_or_default();
        let content = match routing {
            Routing::Auto { .. } => {
                prepared.prepare(crate::source::Needs {
                    pixels: None,
                    coefficients: true,
                })?;
                prepared
                    .coefficients_view()
                    .ok()
                    .as_ref()
                    .and_then(classify_coefficients_view)
            }
            _ => None,
        };
        let predicted = estimate_gain_for(prepared.info(), content);
        let skip = match routing {
            Routing::Always => false,
            Routing::Never => true,
            Routing::Auto { min_gain } => predicted < min_gain,
        };
        let learned = false;
        #[cfg(feature = "chooser")]
        let (predicted, skip, learned) = if routing_override.is_none() && intent == Intent::Fidelity
        {
            match crate::decoded_router::for_source_prepared_with_scratch(
                predicted,
                recorded,
                prepared,
                &mut ws.router_scratch,
            )? {
                Some(decision) => (decision.score() as f32, !decision.restore(), true),
                None => (predicted, skip, learned),
            }
        } else {
            let _ = recorded;
            (predicted, skip, learned)
        };
        #[cfg(not(feature = "chooser"))]
        let _ = recorded;

        // Slack, gate thresholds and deblock policy are derived from the probe
        // and the caller's declarations — never surfaced.
        let mut cfg = RestoreConfig::default()
            .with_threads(threads)
            .with_high_q_identity(!learned)
            .with_projection(Projection::Auto);

        // Conservative gates earlier, trading restoration coverage for fewer
        // observed regressions on the calibration grid. It is not a per-file
        // quality guarantee; alternative routing policies are evaluated separately.
        cfg = cfg.with_high_q_margin(match intent {
            Intent::Fidelity => 0.0,
            Intent::Conservative => 6.0,
        });

        // Provenance currently affects only whether the strict box is
        // available. `FreshEncode` is the one claim that unlocks anything, and
        // even that is left to the family calibration today rather than
        // hard-coded here.
        let _ = provenance;

        // A routed-out file still gets a faithful decode; the caller needs
        // pixels either way, and Unchanged is a first-class outcome.
        if skip {
            cfg = cfg.with_projection(Projection::Off);
        }
        crate::restore_prepared_into(
            model,
            &cfg,
            skip || (require_consistency && geometry_unavailable),
            prepared,
            ws,
        )?;
        let consistency_unavailable =
            geometry_unavailable || ws.restore_report.projection.len() != 3;
        if require_consistency
            && consistency_unavailable
            && !skip
            && !ws.restore_report.skipped_model_high_q
            && !geometry_unavailable
        {
            // A partial projection cannot satisfy the request. Return the
            // decode, never restored pixels labeled Unchanged.
            crate::restore_prepared_into(model, &cfg, true, prepared, ws)?;
        }
        let chroma = match ws.restore_report.chroma_full {
            Some(true) => Chroma::Full,
            Some(false) => Chroma::HalfBoth,
            None => Chroma::Other,
        };
        ws.api_report.encoder_family.clear();
        ws.api_report
            .encoder_family
            .push_str(&ws.restore_report.encoder_family);
        ws.api_report.quality = ws.restore_report.est_quality;
        ws.api_report.quality_scale.clear();
        ws.api_report
            .quality_scale
            .push_str(&ws.restore_report.quality_scale);
        ws.api_report.chroma = chroma;
        ws.api_report.deblocked = ws.restore_report.used_deblock_auto;
        ws.api_report.clamped_fraction.clear();
        ws.api_report
            .clamped_fraction
            .extend(ws.restore_report.projection.iter().map(|p| p.clamped_frac));

        if skip {
            return Ok(OutcomeKind::Unchanged(SkipReason::NotWorthIt {
                estimated_gain: predicted,
            }));
        }
        if ws.restore_report.skipped_model_high_q {
            return Ok(OutcomeKind::Unchanged(SkipReason::NearPristine));
        }
        let consistency = if consistency_unavailable {
            Consistency::NotEnforced(NotCertified::ChromaGeometry)
        } else {
            Consistency::Certified
        };
        if require_consistency && let Consistency::NotEnforced(why) = consistency {
            return Ok(OutcomeKind::Unchanged(SkipReason::ConsistencyUnavailable(
                why,
            )));
        }
        Ok(OutcomeKind::Restored(consistency))
    }

    /// Execute into `ws`, returning borrowed planar pixels and report views
    /// without allocating an owned [`Outcome`].
    ///
    /// When `ws` has already warmed up to the frame/tile dimensions (or was
    /// constructed with [`RepairWorkspace::with_capacity`]) and the caller's
    /// [`Source`](crate::source::Source) reuses its decode buffers (including
    /// [`Source::coefficients_view`](crate::source::Source::coefficients_view)),
    /// engine execution, guards, projection, and strided output writers perform
    /// zero heap allocations.
    ///
    /// (Note: when the `chooser` feature is enabled and default `Intent::Fidelity`
    /// routing runs without an explicit [`Request::routing`] override, the
    /// external `zenanalyze::offer_for_request` feature scanner allocates its
    /// internal `OwnedOffer`.)
    pub fn run_into_workspace<'w>(
        self,
        ws: &'w mut RepairWorkspace,
    ) -> Result<WorkspaceOutcome<'w>, RestoreError> {
        let options = Options {
            intent: self.intent,
            provenance: self.provenance,
            budget: self.budget,
            require_consistency: self.require_consistency,
            routing: self.routing,
        };
        let kind =
            Self::execute_into_workspace(self.restorer, self.source, options, self.threads, ws)?;
        let total = ws.width * ws.height * 3;
        let pixels = PixelsRef {
            planes: &ws.buf_a[..total],
            width: ws.width,
            height: ws.height,
        };
        let report = &ws.api_report;
        Ok(match kind {
            OutcomeKind::Unchanged(why) => WorkspaceOutcome::Unchanged {
                pixels,
                why,
                report,
            },
            OutcomeKind::Restored(consistency) => WorkspaceOutcome::Restored {
                pixels,
                consistency,
                report,
            },
        })
    }

    /// Run it.
    pub fn run(self) -> Result<Outcome, RestoreError> {
        let Self {
            restorer,
            source,
            intent,
            provenance,
            budget,
            routing,
            require_consistency,
            threads,
            workspace,
        } = self;

        let mut temp_ws;
        let (ws, take_planes) = match workspace {
            Some(ws) => (ws, false),
            None => {
                temp_ws = RepairWorkspace::new();
                (&mut temp_ws, true)
            }
        };

        let options = Options {
            intent,
            provenance,
            budget,
            require_consistency,
            routing,
        };
        let kind = Self::execute_into_workspace(restorer, source, options, threads, ws)?;
        let total = ws.width * ws.height * 3;
        let planes = if take_planes {
            ws.buf_a.truncate(total);
            core::mem::take(&mut ws.buf_a)
        } else {
            ws.buf_a[..total].to_vec()
        };
        let pixels = Pixels {
            planes,
            width: ws.width,
            height: ws.height,
        };
        let report = ws.api_report.clone();
        Ok(match kind {
            OutcomeKind::Unchanged(why) => Outcome::Unchanged {
                pixels,
                why,
                report,
            },
            OutcomeKind::Restored(consistency) => Outcome::Restored {
                pixels,
                consistency,
                report,
            },
        })
    }
}
