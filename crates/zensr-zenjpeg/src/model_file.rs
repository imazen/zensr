//! Versioned, weight-bound runtime model files. See docs/MODEL_CONTAINER.md.
use crate::api::{Budget, Content, Restorer};
use crate::{AdoptedModel, GuardConfig, ModelSpace, ProjectionConfig};
use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::Path;
use std::sync::Arc;

const MAGIC: &[u8; 8] = b"ZSRMODEL";
const LIMIT: usize = 64 * 1024 * 1024;
const HEADER: usize = 64;

/// A model, its input interpretation and its weight-bound runtime policy.
///
/// Loading validates the complete file before unpacking weights. The existing
/// CPU-dispatched model kernels pack weights once; reuse the loaded object.
pub struct ModelBundle {
    name: String,
    model: AdoptedModel,
    budget: Budget,
    pub(crate) policy: Arc<RuntimePolicy>,
}

impl ModelBundle {
    /// Load an entire `.zensr` file. Unknown versions, mismatched calibration,
    /// malformed/non-finite data and unsupported feature identities are errors.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < HEADER || bytes.len() > LIMIT || &bytes[..8] != MAGIC {
            return Err("invalid model container header or size".into());
        }
        let mut header = Reader(&bytes[8..32]);
        if header.u32()? != 1 {
            return Err("unsupported model container version".into());
        }
        let meta_len = header.u32()? as usize;
        let policy_len = usize::try_from(header.u64()?).map_err(|_| "policy size overflow")?;
        let weights_len = usize::try_from(header.u64()?).map_err(|_| "weights size overflow")?;
        let total = HEADER
            .checked_add(meta_len)
            .and_then(|n| n.checked_add(policy_len))
            .and_then(|n| n.checked_add(weights_len));
        if total != Some(bytes.len()) || meta_len > 4096 || policy_len > 64 * 1024 {
            return Err("invalid model container section lengths".into());
        }
        let mut digest = Sha256::new();
        digest.update(&bytes[..32]);
        digest.update(&bytes[HEADER..]);
        if digest.finalize().as_slice() != &bytes[32..HEADER] {
            return Err("model container checksum mismatch".into());
        }
        let (meta_bytes, rest) = bytes[HEADER..].split_at(meta_len);
        let (policy_bytes, weights) = rest.split_at(policy_len);
        let mut meta = Reader(meta_bytes);
        let name = meta.string(256)?;
        let arch = meta.u8()?;
        let space = meta.u8()?;
        let scale = meta.u8()? as usize;
        let nf = meta.u16()? as usize;
        let nc = meta.u16()? as usize;
        let budget = match meta.u8()? {
            0 => Budget::Realtime,
            1 => Budget::Quality,
            _ => return Err("invalid model budget".into()),
        };
        let encoding = meta.u8()?;
        meta.end()?;
        if !matches!(scale, 1 | 2 | 4) || nf == 0 || nf > 256 || nc > 128 || space > 1 {
            return Err("unsupported model geometry or colour space".into());
        }
        let word = match encoding {
            1 => 2,
            2 => 4,
            _ => return Err("unknown weight encoding".into()),
        };
        if !weights_len.is_multiple_of(word) {
            return Err("partial weight value".into());
        }
        let count = match arch {
            1 if (4..=64).contains(&nf) && nf.is_multiple_of(4) => {
                29 * nf + nc * (9 * nf * nf + 2 * nf) + 27 * nf * scale * scale + 3 * scale * scale
            }
            1 => return Err("unsupported Compact channel geometry".into()),
            2 if nf == 48 && nc == 6 && scale != 1 => {
                28 * nf
                    + 19 * (9 * nf * nf + nf)
                    + 4 * nf * nf
                    + nf
                    + 27 * nf * scale * scale
                    + 3 * scale * scale
            }
            2 => return Err("Span48 requires nf=48, nc=6 and scale=2 or 4".into()),
            _ => return Err("unknown model architecture".into()),
        };
        if count.checked_mul(word) != Some(weights_len) {
            return Err("weights do not match model geometry".into());
        }
        let policy = RuntimePolicy::read(policy_bytes, meta_bytes, weights)?;
        let raw = if encoding == 1 {
            zensr_micro::decode_all_f16(weights)
        } else {
            weights
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect()
        };
        if !raw.iter().all(|v| v.is_finite()) {
            return Err("non-finite model weights".into());
        }
        let mut model = if arch == 1 {
            AdoptedModel::load_compact(&raw, nf, nc, scale)?
        } else {
            AdoptedModel::load_span48(&raw, scale)?
        };
        if space == 1 {
            model.set_space(ModelSpace::Ycbcr);
        }
        Ok(Self {
            name,
            model,
            budget,
            policy: Arc::new(policy),
        })
    }

    /// Read a bounded model file; the same validation is used for embedded bytes.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, String> {
        let file = std::fs::File::open(path).map_err(|e| format!("open model: {e}"))?;
        let size = file
            .metadata()
            .map_err(|e| format!("model metadata: {e}"))?
            .len();
        if size > LIMIT as u64 {
            return Err("model file exceeds size limit".into());
        }
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(size as usize)
            .map_err(|_| "model allocation failed")?;
        file.take(LIMIT as u64 + 1)
            .read_to_end(&mut bytes)
            .map_err(|e| format!("read model: {e}"))?;
        Self::from_bytes(&bytes)
    }

    /// The artifact's human-readable model name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The loaded CPU-dispatched model, also usable for separate ×2/×4 SR.
    pub fn model(&self) -> &AdoptedModel {
        &self.model
    }

    /// Construct a reusable JPEG restorer with this file's model and policy.
    /// JPEG restoration requires a ×1 model. Uncalibrated files require an
    /// explicit `Routing::Always` or `Routing::Never` request.
    pub fn into_restorer(self) -> Result<Restorer, String> {
        if self.model.scale != 1 {
            return Err("JPEG restoration requires a x1 model".into());
        }
        Ok(Restorer::from_bundle(self.model, self.budget, self.policy))
    }
}

pub(crate) struct RuntimePolicy {
    pub mode: u8,
    pub tile: usize,
    pub deblock_max_quality: f32,
    pub guard: GuardConfig,
    slack: [f32; 6],
    pub min_gain: f32,
    pub conservative_margin: f32,
    pub graphic_threshold: f32,
    pub gates: [f32; 6],
    pub curves: [Vec<(f32, f32)>; 14],
    // Decoded files fail explicitly without chooser; keep one bounded parser.
    #[cfg_attr(not(feature = "chooser"), allow(dead_code))]
    pub router: Option<Router>,
}

#[cfg_attr(not(feature = "chooser"), allow(dead_code))]
pub(crate) struct Router {
    pub mean: [f64; 37],
    pub scale: [f64; 37],
    pub gain: [f64; 47],
    pub risk: [f64; 47],
    pub gain_bias: f64,
    pub risk_bias: f64,
    pub penalty: f64,
    pub threshold: f64,
    pub categories: [Vec<String>; 3],
}

impl RuntimePolicy {
    fn read(bytes: &[u8], metadata: &[u8], weights: &[u8]) -> Result<Self, String> {
        let mut r = Reader(bytes);
        let mode = r.u8()?;
        if mode > 2 || r.u32()? != 1 {
            return Err("unsupported runtime pipeline".into());
        }
        let tile = r.u32()? as usize;
        let deblock_max_quality = r.f32()?;
        let mut guard = GuardConfig::default();
        guard.residual_clamp = r.f32()?;
        guard.texture_gate = match r.u8()? {
            0 => false,
            1 => true,
            _ => return Err("invalid guard flag".into()),
        };
        guard.texture_alpha_min = r.f32()?;
        guard.texture_k = r.f32()?;
        guard.roundtrip_mae_max = r.f32()?;
        let slack = r.f32s::<6>()?;
        let min_gain = r.f32()?;
        let conservative_margin = r.f32()?;
        if tile > 4096
            || !(0.0..=100.0).contains(&deblock_max_quality)
            || guard.residual_clamp < 0.0
            || !(0.0..=1.0).contains(&guard.texture_alpha_min)
            || guard.texture_k <= 0.0
            || guard.roundtrip_mae_max <= 0.0
            || slack.iter().any(|v| *v < 0.0)
            || conservative_margin < 0.0
        {
            return Err("invalid runtime policy bounds".into());
        }
        let bound: [u8; 32] = r.take(32)?.try_into().unwrap();
        let mut model_identity = Sha256::new();
        model_identity.update(metadata);
        model_identity.update(weights);
        if mode != 0 && bound.as_slice() != model_identity.finalize().as_slice() {
            return Err(
                "routing calibration belongs to different model metadata or weights".into(),
            );
        }
        let mut graphic_threshold = 0.0;
        let mut gates = [0.; 6];
        let mut curves: [Vec<(f32, f32)>; 14] = std::array::from_fn(|_| Vec::new());
        if mode != 0 {
            graphic_threshold = r.f32()?;
            gates = r.f32s()?;
            if !(0.0..=1.0).contains(&graphic_threshold) || gates.iter().any(|v| *v < 0.0) {
                return Err("invalid calibrated gate".into());
            }
            for curve in &mut curves {
                let count = r.u16()? as usize;
                if !(2..=256).contains(&count) {
                    return Err("invalid gain curve size".into());
                }
                for _ in 0..count {
                    curve.push((r.f32()?, r.f32()?));
                }
                if curve.windows(2).any(|w| w[0].0 >= w[1].0) {
                    return Err("unsorted gain curve".into());
                }
            }
        }
        let router = if mode == 1 {
            if !cfg!(feature = "chooser") {
                return Err("model router requires chooser support".into());
            }
            let schema = r.take(32)?;
            #[cfg(feature = "chooser")]
            if schema != Sha256::digest(include_bytes!("decoded_router_features.rs")).as_slice() {
                return Err("unsupported decoded feature schema".into());
            }
            #[cfg(not(feature = "chooser"))]
            let _ = schema;
            let mean = r.f64s()?;
            let scale = r.f64s::<37>()?;
            let gain = r.f64s()?;
            let risk = r.f64s()?;
            let gain_bias = r.f64()?;
            let risk_bias = r.f64()?;
            let penalty = r.f64()?;
            let threshold = r.f64()?;
            let mut categories: [Vec<String>; 3] = std::array::from_fn(|_| Vec::new());
            for group in &mut categories {
                let count = r.u8()?;
                if count == 0 || count > 32 {
                    return Err("invalid router category count".into());
                }
                for _ in 0..count {
                    group.push(r.string(128)?);
                }
                for (i, value) in group.iter().enumerate() {
                    if group[..i].contains(value) {
                        return Err("duplicate router category".into());
                    }
                }
            }
            if categories.iter().map(Vec::len).sum::<usize>() != 10
                || scale.iter().any(|v| *v <= 0.0)
                || penalty < 0.0
            {
                return Err("invalid router dimensions or scale".into());
            }
            Some(Router {
                mean,
                scale,
                gain,
                risk,
                gain_bias,
                risk_bias,
                penalty,
                threshold,
                categories,
            })
        } else {
            None
        };
        r.end()?;
        Ok(Self {
            mode,
            tile,
            deblock_max_quality,
            guard,
            slack,
            min_gain,
            conservative_margin,
            graphic_threshold,
            gates,
            curves,
            router,
        })
    }

    pub fn projection(&self, probe: &zenjpeg::detect::JpegProbe) -> ProjectionConfig {
        let family = format!("{:?}", probe.encoder);
        let relative = if family.starts_with("Cjpegli") {
            self.slack[2]
        } else if family == "Mozjpeg" {
            self.slack[1]
        } else if crate::is_round_to_nearest_family(&family) {
            self.slack[0]
        } else {
            self.slack[3]
        };
        let absolute = if family.starts_with("Cjpegli") {
            self.slack[4]
        } else {
            self.slack[5]
        };
        ProjectionConfig::with_slack_q(relative).with_slack_abs(absolute)
    }

    pub fn gain(&self, probe: &zenjpeg::detect::JpegProbe, content: Option<Content>) -> f32 {
        crate::api::estimate_gain_with_curves(
            probe,
            content,
            std::array::from_fn(|i| self.curves[i].as_slice()),
        )
    }
}

struct Reader<'a>(&'a [u8]);
impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let (value, rest) = self
            .0
            .split_at_checked(n)
            .ok_or("truncated model container")?;
        self.0 = rest;
        Ok(value)
    }
    fn u8(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16, String> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn f32(&mut self) -> Result<f32, String> {
        let value = f32::from_bits(self.u32()?);
        value
            .is_finite()
            .then_some(value)
            .ok_or_else(|| "non-finite policy value".into())
    }
    fn f64(&mut self) -> Result<f64, String> {
        let value = f64::from_bits(self.u64()?);
        value
            .is_finite()
            .then_some(value)
            .ok_or_else(|| "non-finite router value".into())
    }
    fn f32s<const N: usize>(&mut self) -> Result<[f32; N], String> {
        let mut a = [0.; N];
        for v in &mut a {
            *v = self.f32()?;
        }
        Ok(a)
    }
    fn f64s<const N: usize>(&mut self) -> Result<[f64; N], String> {
        let mut a = [0.; N];
        for v in &mut a {
            *v = self.f64()?;
        }
        Ok(a)
    }
    fn string(&mut self, limit: usize) -> Result<String, String> {
        let n = self.u16()? as usize;
        if n == 0 || n > limit {
            return Err("invalid model string length".into());
        }
        let s = std::str::from_utf8(self.take(n)?).map_err(|_| "invalid model UTF-8")?;
        if s.contains('\0') {
            return Err("model strings cannot contain NUL".into());
        }
        Ok(s.to_owned())
    }
    fn end(self) -> Result<(), String> {
        if self.0.is_empty() {
            Ok(())
        } else {
            Err("unexpected model section bytes".into())
        }
    }
}

#[cfg(test)]
#[path = "model_file_tests.rs"]
pub(crate) mod tests;
