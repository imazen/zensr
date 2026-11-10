//! Shared strided resize/blend operations for resident offline studies.
use std::error::Error;
use zenresize::{Filter, PixelDescriptor, ResizeConfig, Resizer, TransferFunction};
type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;
#[derive(Clone, Copy)]
pub(crate) struct View<'a, T> {
    pub(crate) data: &'a [T],
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) stride: usize,
}
impl<T> View<'_, T> {
    fn validate(&self) -> Result<()> {
        if self.width == 0
            || self.height == 0
            || self.stride < self.width * 3
            || self.data.len() < (self.height - 1) * self.stride + self.width * 3
        {
            return Err("Invalid RGB row geometry".into());
        }
        Ok(())
    }
}
pub(crate) fn filter(name: &str) -> Result<Filter> {
    Ok(match name {
        "triangle" => Filter::Triangle,
        "mitchell" => Filter::Mitchell,
        "catmull_rom" => Filter::CatmullRom,
        "lanczos2" => Filter::Lanczos2,
        _ => return Err("Unknown filter".into()),
    })
}
pub(crate) fn float_resize(
    source: View<'_, f32>,
    w: usize,
    h: usize,
    kernel: Filter,
) -> Result<Vec<f32>> {
    source.validate()?;
    let config = ResizeConfig::builder(
        source.width as u32,
        source.height as u32,
        w as u32,
        h as u32,
    )
    .format(PixelDescriptor::RGBF32_LINEAR.with_transfer(TransferFunction::Srgb))
    .filter(kernel)
    .srgb()
    .in_stride(source.stride)
    .build();
    Ok(Resizer::new(&config).resize_f32(source.data))
}
pub(crate) fn rgb_view(data: &[u8], w: usize, h: usize) -> View<'_, u8> {
    View {
        data,
        width: w,
        height: h,
        stride: w * 3,
    }
}
pub(crate) fn floats(source: View<'_, u8>) -> Result<Vec<f32>> {
    source.validate()?;
    let mut data = Vec::with_capacity(source.width * source.height * 3);
    for y in 0..source.height {
        data.extend(
            source.data[y * source.stride..y * source.stride + source.width * 3]
                .iter()
                .map(|&v| f32::from(v) / 255.0),
        );
    }
    Ok(data)
}
pub(crate) fn blend(
    base: View<'_, f32>,
    sr: View<'_, f32>,
    quarter: u8,
    out: &mut Vec<u8>,
) -> Result<()> {
    base.validate()?;
    sr.validate()?;
    if (base.width, base.height) != (sr.width, sr.height) || quarter > 4 {
        return Err("Blend shape or intensity differs".into());
    }
    out.clear();
    out.reserve(base.width * base.height * 3);
    for y in 0..base.height {
        for x in 0..base.width * 3 {
            let a = base.data[y * base.stride + x];
            let b = sr.data[y * sr.stride + x];
            let v = match quarter {
                0 => a,
                4 => b,
                q => a * (f32::from(4 - q) * 0.25) + b * (f32::from(q) * 0.25),
            };
            if !v.is_finite() {
                return Err("Nonfinite blend pixels".into());
            }
            out.push((v * 255.0 + 0.5).clamp(0.0, 255.0) as u8);
        }
    }
    Ok(())
}
