//! Per-request decode reuse. Never reuse Off pixels for an Auto decode.
use crate::RestoreError;
use zenjpeg::decoder::{DecodeResult, DecodedCoefficients, Decoder};

#[derive(Default)]
pub(crate) struct Prepared {
    pub(crate) coefficients: Option<DecodedCoefficients>,
    pub(crate) plain: Option<DecodeResult>,
}

impl Prepared {
    pub(crate) fn coefficients(
        &mut self,
        data: &[u8],
    ) -> Result<&DecodedCoefficients, RestoreError> {
        if self.coefficients.is_none() {
            self.coefficients = Some(
                Decoder::new()
                    .decode_coefficients(data, enough::Unstoppable)
                    .map_err(|e| RestoreError::Decode(format!("coefficients: {e:?}")))?,
            );
        }
        Ok(self.coefficients.as_ref().unwrap())
    }

    pub(crate) fn plain(&mut self, data: &[u8]) -> Result<&DecodeResult, RestoreError> {
        if self.plain.is_none() {
            self.plain = Some(
                Decoder::new()
                    .deblock(zenjpeg::decoder::DeblockMode::Off)
                    .decode(data, enough::Unstoppable)
                    .map_err(|e| RestoreError::Decode(format!("{e:?}")))?,
            );
        }
        Ok(self.plain.as_ref().unwrap())
    }
}
