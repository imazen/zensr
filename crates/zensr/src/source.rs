//! Codec-supplied samples and coefficient views. This crate never decodes JPEG.
use crate::RestoreError;
use std::fmt;

/// Stable codec classification spelling; Debug intentionally preserves the measured category.
#[derive(Clone, PartialEq, Eq)]
pub struct Category(pub String);
impl fmt::Debug for Category {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
#[derive(Clone, Copy, Debug)]
pub struct Dimensions {
    pub width: u32,
    pub height: u32,
}
#[derive(Clone, Debug)]
pub struct Quality {
    pub value: f32,
    pub scale: Category,
}
/// Codec fingerprint after applying any authenticated encoder metadata.
#[derive(Clone, Debug)]
pub struct JpegInfo {
    pub dimensions: Dimensions,
    pub encoder: Category,
    pub subsampling: Category,
    pub quality: Quality,
    pub num_components: u8,
    pub recorded: bool,
    pub jpegli_aq_off: bool,
    pub encoded_len: usize,
}
/// Component coefficients in JPEG zigzag order; no coefficient samples are copied.
#[derive(Clone, Copy, Debug)]
pub struct Component<'a> {
    pub coeffs: &'a [i16],
    pub blocks_wide: usize,
    pub blocks_high: usize,
    pub quant_table_idx: u8,
    pub h_samp: u8,
    pub v_samp: u8,
}
#[derive(Debug)]
pub struct Coefficients<'a> {
    pub components: Vec<Component<'a>>,
    pub quant_tables: &'a [Option<[u16; 64]>],
}
/// Interleaved RGB8 in the JPEG's coding color space, before ICC conversion.
#[derive(Clone, Copy, Debug)]
pub struct EncodedRgb8<'a> {
    pub data: &'a [u8],
    pub width: u32,
    pub height: u32,
    pub stride: usize,
}
impl EncodedRgb8<'_> {
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodeMode {
    Plain,
    Deblocked,
}
#[derive(Clone, Copy, Debug)]
pub struct Needs {
    pub pixels: Option<DecodeMode>,
    pub coefficients: bool,
}
/// The codec owns parsing, allocation limits and decoding. The engine owns all repair policy.
pub trait Source {
    fn info(&self) -> &JpegInfo;
    fn prepare(&mut self, needs: Needs) -> Result<(), RestoreError>;
    fn pixels(&self, mode: DecodeMode) -> Result<EncodedRgb8<'_>, RestoreError>;
    fn coefficients(&self) -> Result<Coefficients<'_>, RestoreError>;
}
