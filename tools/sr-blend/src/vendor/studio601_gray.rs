use crate::decode::Rgb8Image;
pub fn studio601_gray(img: &Rgb8Image) -> Rgb8Image {
    let w = (img.width & !1) as usize;
    let h = (img.height & !1) as usize;
    let src_w = img.width as usize;
    let mut pixels = vec![0u8; w * h * 3];
    let fill = |orow: &mut [u8], srow: &[u8]| {
        for (o, px) in orow
            .as_chunks_mut::<3>()
            .0
            .iter_mut()
            .zip(srow.as_chunks::<3>().0)
        {
            let yv = (16.0
                + (65.481 * px[0] as f32 + 128.553 * px[1] as f32 + 24.966 * px[2] as f32) / 255.0)
                .round()
                .clamp(0.0, 255.0) as u8;
            o.copy_from_slice(&[yv, yv, yv]);
        }
    };
    if w * h >= (1 << 20) && rayon::current_num_threads() > 1 {
        use rayon::prelude::*;
        pixels
            .par_chunks_mut(w * 3)
            .enumerate()
            .for_each(|(y, orow)| fill(orow, &img.pixels[y * src_w * 3..y * src_w * 3 + w * 3]));
    } else {
        pixels
            .chunks_mut(w * 3)
            .enumerate()
            .for_each(|(y, orow)| fill(orow, &img.pixels[y * src_w * 3..y * src_w * 3 + w * 3]));
    }
    Rgb8Image {
        pixels,
        width: w as u32,
        height: h as u32,
    }
}
