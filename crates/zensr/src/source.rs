//! Codec-supplied samples and coefficient views. This crate never decodes JPEG.
use crate::RestoreError;
use std::fmt;

/// Stable codec classification spelling; Debug intentionally preserves the measured category.
#[derive(Clone, PartialEq, Eq)]
pub struct Category(pub String);
impl Category {
    /// Borrow the underlying classification string without formatting overhead.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}
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

/// Borrowed 8×8 JPEG DCT coefficient grid in standard 64-element zigzag order.
///
/// Matches the zero-copy `(coeffs: &[i16], blocks_wide, blocks_high)` layout
/// exposed by `zenjpeg` (`ComponentCoefficients`) and `zensally`
/// (`zensally::JpegDctBlocks`), so a single decoded coefficient plane can feed
/// both restoration projection and smart-crop AC energy folding without copying
/// or reordering coefficients.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JpegDctBlocks<'a> {
    /// Quantized (or dequantized) 8×8 block coefficients in JPEG zigzag order:
    /// `blocks_wide * blocks_high * 64` `i16` elements, row-major by block.
    /// Within each 64-element block slice `b`, `b[0]` is DC and `b[1..64]` are
    /// the 63 AC coefficients in zigzag order.
    pub coeffs: &'a [i16],
    /// Number of 8×8 blocks horizontally.
    pub blocks_wide: usize,
    /// Number of 8×8 blocks vertically.
    pub blocks_high: usize,
    /// Optional 64-element quantization table in JPEG zigzag order (matching
    /// `coeffs`). Note that [`Coefficients::quant_tables`] stores DQT tables in
    /// 8×8 natural (row-major) order; use [`Coefficients::quant_table_zigzag`]
    /// when converting a DQT table to zigzag order.
    pub quant_table: Option<&'a [u16; 64]>,
}

impl<'a> JpegDctBlocks<'a> {
    /// Construct a borrowed JPEG 8×8 zigzag DCT block grid, validating that
    /// `coeffs.len() >= blocks_wide * blocks_high * 64`.
    pub fn new(
        coeffs: &'a [i16],
        blocks_wide: usize,
        blocks_high: usize,
    ) -> Result<Self, &'static str> {
        let needed = blocks_wide
            .checked_mul(blocks_high)
            .and_then(|n| n.checked_mul(64))
            .ok_or("JPEG block grid dimensions overflow")?;
        if coeffs.len() < needed {
            return Err("JPEG DCT coefficient slice too small for blocks_wide * blocks_high * 64");
        }
        Ok(Self {
            coeffs,
            blocks_wide,
            blocks_high,
            quant_table: None,
        })
    }

    /// Attach a 64-element zigzag quantization table.
    #[must_use]
    pub fn with_quant_table(mut self, quant_table: &'a [u16; 64]) -> Self {
        self.quant_table = Some(quant_table);
        self
    }
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

impl<'a> Component<'a> {
    /// Construct a validated component coefficient view in JPEG zigzag order.
    pub fn new(
        coeffs: &'a [i16],
        blocks_wide: usize,
        blocks_high: usize,
        quant_table_idx: u8,
        h_samp: u8,
        v_samp: u8,
    ) -> Result<Self, &'static str> {
        let blocks = JpegDctBlocks::new(coeffs, blocks_wide, blocks_high)?;
        Ok(Self::from_dct_blocks(
            blocks,
            quant_table_idx,
            h_samp,
            v_samp,
        ))
    }

    /// Wrap a validated [`JpegDctBlocks`] grid with JPEG component metadata.
    #[must_use]
    pub fn from_dct_blocks(
        blocks: JpegDctBlocks<'a>,
        quant_table_idx: u8,
        h_samp: u8,
        v_samp: u8,
    ) -> Self {
        Self {
            coeffs: blocks.coeffs,
            blocks_wide: blocks.blocks_wide,
            blocks_high: blocks.blocks_high,
            quant_table_idx,
            h_samp,
            v_samp,
        }
    }

    /// Borrow this component's 8×8 zigzag coefficient grid as [`JpegDctBlocks`].
    pub fn dct_blocks(&self) -> Result<JpegDctBlocks<'a>, &'static str> {
        JpegDctBlocks::new(self.coeffs, self.blocks_wide, self.blocks_high)
    }
}

#[derive(Debug)]
pub struct Coefficients<'a> {
    pub components: Vec<Component<'a>>,
    /// 8×8 quantization tables in natural (row-major) order, indexed by
    /// [`Component::quant_table_idx`].
    pub quant_tables: &'a [Option<[u16; 64]>],
}

impl<'a> Coefficients<'a> {
    /// Borrow quantization table `idx` in 8×8 natural (row-major) order.
    #[must_use]
    pub fn quant_table_natural(&self, idx: u8) -> Option<&'a [u16; 64]> {
        self.quant_tables
            .get(usize::from(idx))
            .and_then(Option::as_ref)
    }

    /// Return quantization table `idx` permuted into 64-element JPEG zigzag
    /// order (matching [`Component::coeffs`] and [`JpegDctBlocks::coeffs`]).
    #[must_use]
    pub fn quant_table_zigzag(&self, idx: u8) -> Option<[u16; 64]> {
        let qt = self.quant_table_natural(idx)?;
        Some(std::array::from_fn(|k| {
            qt[zensr_micro::consist::ZIGZAG_TO_NATURAL[k]]
        }))
    }
}

/// Stack-borrowed view of up to 4 JPEG component coefficient planes and DQT
/// tables, avoiding the per-call `Vec` allocation of [`Coefficients`] on warm
/// [`RepairWorkspace`](crate::api::RepairWorkspace) execution paths.
#[derive(Clone, Copy, Debug)]
pub struct CoefficientsView<'a> {
    components: [Option<Component<'a>>; 4],
    stored: usize,
    total_components: usize,
    /// 8×8 quantization tables in natural (row-major) order.
    pub quant_tables: &'a [Option<[u16; 64]>],
}

impl<'a> CoefficientsView<'a> {
    /// Construct a zero-allocation stack view from a borrowed slice of components.
    #[must_use]
    pub fn new(components: &[Component<'a>], quant_tables: &'a [Option<[u16; 64]>]) -> Self {
        let mut arr = [None; 4];
        let stored = components.len().min(4);
        for (dst, &src) in arr[..stored].iter_mut().zip(&components[..stored]) {
            *dst = Some(src);
        }
        Self {
            components: arr,
            stored,
            total_components: components.len(),
            quant_tables,
        }
    }

    /// Construct from an owned-`Vec` [`Coefficients`] container.
    #[must_use]
    pub fn from_coefficients(coeffs: &Coefficients<'a>) -> Self {
        Self::new(&coeffs.components, coeffs.quant_tables)
    }

    /// Total number of components reported by the codec (including any past 4).
    #[must_use]
    pub fn total_components(&self) -> usize {
        self.total_components
    }

    /// Number of inline components stored in this view (`<= 4`).
    #[must_use]
    pub fn len(&self) -> usize {
        self.stored
    }

    /// True when the view holds zero components.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.stored == 0
    }

    /// Get inline component `idx` (`0..4`).
    #[must_use]
    pub fn get(&self, idx: usize) -> Option<&Component<'a>> {
        self.components.get(idx).and_then(Option::as_ref)
    }

    /// Borrow the first (luma) component if present.
    #[must_use]
    pub fn first(&self) -> Option<&Component<'a>> {
        self.get(0)
    }

    /// Iterate over the inline borrowed components (`0..self.len()`).
    pub fn iter(&self) -> impl Iterator<Item = &Component<'a>> {
        self.components[..self.stored].iter().flatten()
    }

    /// Borrow quantization table `idx` in 8×8 natural (row-major) order.
    #[must_use]
    pub fn quant_table_natural(&self, idx: u8) -> Option<&'a [u16; 64]> {
        self.quant_tables
            .get(usize::from(idx))
            .and_then(Option::as_ref)
    }

    /// Return quantization table `idx` permuted into 64-element JPEG zigzag order.
    #[must_use]
    pub fn quant_table_zigzag(&self, idx: u8) -> Option<[u16; 64]> {
        let qt = self.quant_table_natural(idx)?;
        Some(std::array::from_fn(|k| {
            qt[zensr_micro::consist::ZIGZAG_TO_NATURAL[k]]
        }))
    }
}

/// Interleaved RGB8 in the JPEG's coding color space, before ICC conversion.
#[derive(Clone, Copy, Debug)]
pub struct EncodedRgb8<'a> {
    pub data: &'a [u8],
    pub width: u32,
    pub height: u32,
    pub stride: usize,
}
impl<'a> EncodedRgb8<'a> {
    /// Construct a tightly packed (`stride = width * 3`) RGB8 view, validating
    /// dimensions and slice length.
    pub fn new(data: &'a [u8], width: u32, height: u32) -> Result<Self, &'static str> {
        let stride = (width as usize)
            .checked_mul(3)
            .ok_or("RGB8 row width overflow")?;
        Self::new_strided(data, width, height, stride)
    }

    /// Construct a strided RGB8 view (`stride >= width * 3`), validating that
    /// `data` covers all `height` rows (allowing an unpadded final row).
    pub fn new_strided(
        data: &'a [u8],
        width: u32,
        height: u32,
        stride: usize,
    ) -> Result<Self, &'static str> {
        let row_bytes = (width as usize)
            .checked_mul(3)
            .ok_or("RGB8 row width overflow")?;
        if stride < row_bytes {
            return Err("row stride smaller than width * 3");
        }
        let expected = if height == 0 {
            0
        } else {
            (height as usize - 1)
                .checked_mul(stride)
                .and_then(|n| n.checked_add(row_bytes))
                .ok_or("RGB8 strided buffer size overflow")?
        };
        if data.len() < expected {
            return Err("RGB8 slice too small for given dimensions and stride");
        }
        Ok(Self {
            data,
            width,
            height,
            stride,
        })
    }

    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Borrow any 8-bit 3-channel RGB [`zenpixels::PixelSlice`] as a zero-copy
    /// strided [`EncodedRgb8`].
    ///
    /// Note that `zensr` operates in the JPEG's coding RGB space before ICC
    /// conversion; any colorimetry descriptor attached to `slice` is the
    /// caller/codec's responsibility to preserve across restoration.
    #[cfg(feature = "zenpixels")]
    pub fn from_pixel_slice<P>(slice: zenpixels::PixelSlice<'a, P>) -> Result<Self, &'static str> {
        let desc = slice.descriptor();
        if desc.channel_type() != zenpixels::ChannelType::U8
            || desc.layout() != zenpixels::ChannelLayout::Rgb
        {
            return Err("EncodedRgb8 requires an 8-bit 3-channel RGB PixelSlice");
        }
        Ok(Self {
            data: slice.as_strided_bytes(),
            width: slice.width(),
            height: slice.rows(),
            stride: slice.stride(),
        })
    }
}

#[cfg(feature = "zenpixels")]
impl<'a> From<zenpixels::PixelSlice<'a, rgb::Rgb<u8>>> for EncodedRgb8<'a> {
    fn from(slice: zenpixels::PixelSlice<'a, rgb::Rgb<u8>>) -> Self {
        Self {
            data: slice.as_strided_bytes(),
            width: slice.width(),
            height: slice.rows(),
            stride: slice.stride(),
        }
    }
}

#[cfg(feature = "zenpixels")]
impl<'a> TryFrom<zenpixels::PixelSlice<'a>> for EncodedRgb8<'a> {
    type Error = &'static str;

    fn try_from(slice: zenpixels::PixelSlice<'a>) -> Result<Self, Self::Error> {
        Self::from_pixel_slice(slice)
    }
}

/// Converts [`EncodedRgb8`] into a type-erased [`zenpixels::PixelSlice`] tagged
/// with [`zenpixels::PixelDescriptor::RGB8`] (`TransferFunction::Unknown`),
/// reflecting that [`EncodedRgb8`] holds coding-space RGB samples prior to ICC
/// conversion rather than guaranteed sRGB.
#[cfg(feature = "zenpixels")]
impl<'a> TryFrom<EncodedRgb8<'a>> for zenpixels::PixelSlice<'a> {
    type Error = zenpixels::BufferError;

    fn try_from(rgb: EncodedRgb8<'a>) -> Result<Self, Self::Error> {
        zenpixels::PixelSlice::new(
            rgb.data,
            rgb.width,
            rgb.height,
            rgb.stride,
            zenpixels::PixelDescriptor::RGB8,
        )
        .map_err(|e| e.decompose().0)
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

    /// Return a stack-borrowed view of up to 4 component coefficient planes
    /// without allocating a per-call `Vec` when overridden by the codec.
    ///
    /// The default implementation delegates to [`Source::coefficients`] for
    /// full backwards compatibility with existing `Source` implementations.
    fn coefficients_view(&self) -> Result<CoefficientsView<'_>, RestoreError> {
        let coeffs = self.coefficients()?;
        Ok(CoefficientsView::from_coefficients(&coeffs))
    }
}
