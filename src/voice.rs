use crate::{Checkpoint, Error, Result, Value};

/// Named form of the positional generator configuration stored in common
/// real-time voice-conversion checkpoints.
#[derive(Clone, Debug, PartialEq)]
pub struct VoiceModelConfig {
    pub spectrogram_channels: u32,
    pub segment_size: u32,
    pub intermediate_channels: u32,
    pub hidden_channels: u32,
    pub filter_channels: u32,
    pub attention_heads: u32,
    pub attention_layers: u32,
    pub kernel_size: u32,
    pub dropout: f64,
    pub resblock: String,
    pub resblock_kernel_sizes: Vec<u32>,
    pub resblock_dilation_sizes: Vec<Vec<u32>>,
    pub upsample_rates: Vec<u32>,
    pub upsample_initial_channels: u32,
    pub upsample_kernel_sizes: Vec<u32>,
    pub speaker_count: u32,
    pub speaker_embedding_channels: u32,
    pub sample_rate: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VoiceModelInfo {
    pub config: VoiceModelConfig,
    pub architecture_version: Option<String>,
    pub sample_rate_label: Option<String>,
    pub pitch_guidance: bool,
    pub training_info: Option<String>,
    pub phone_feature_channels: Option<u32>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ValidationReport {
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}

impl ValidationReport {
    pub fn is_valid(&self) -> bool {
        self.errors.is_empty()
    }
}

impl Checkpoint {
    pub fn voice_model_info(&self) -> Result<VoiceModelInfo> {
        VoiceModelInfo::from_checkpoint(self)
    }
}

impl VoiceModelInfo {
    pub fn from_checkpoint(checkpoint: &Checkpoint) -> Result<Self> {
        let config_value = checkpoint
            .get("config")
            .ok_or_else(|| Error::InvalidArchive("voice model config is missing".into()))?;
        let values = config_value
            .as_list()
            .ok_or_else(|| Error::InvalidArchive("voice model config is not a list".into()))?;
        if values.len() < 18 {
            return Err(Error::InvalidArchive(format!(
                "voice model config has {} fields; expected at least 18",
                values.len()
            )));
        }
        let config = VoiceModelConfig {
            spectrogram_channels: uint(&values[0], "spectrogram_channels")?,
            segment_size: uint(&values[1], "segment_size")?,
            intermediate_channels: uint(&values[2], "intermediate_channels")?,
            hidden_channels: uint(&values[3], "hidden_channels")?,
            filter_channels: uint(&values[4], "filter_channels")?,
            attention_heads: uint(&values[5], "attention_heads")?,
            attention_layers: uint(&values[6], "attention_layers")?,
            kernel_size: uint(&values[7], "kernel_size")?,
            dropout: number(&values[8], "dropout")?,
            resblock: string(&values[9], "resblock")?,
            resblock_kernel_sizes: uint_list(&values[10], "resblock_kernel_sizes")?,
            resblock_dilation_sizes: nested_uint_list(&values[11], "resblock_dilation_sizes")?,
            upsample_rates: uint_list(&values[12], "upsample_rates")?,
            upsample_initial_channels: uint(&values[13], "upsample_initial_channels")?,
            upsample_kernel_sizes: uint_list(&values[14], "upsample_kernel_sizes")?,
            speaker_count: uint(&values[15], "speaker_count")?,
            speaker_embedding_channels: uint(&values[16], "speaker_embedding_channels")?,
            sample_rate: uint(&values[17], "sample_rate")?,
        };
        let phone_feature_channels = checkpoint
            .tensor("enc_p.emb_phone.weight")
            .and_then(|tensor| tensor.shape.get(1))
            .and_then(|value| u32::try_from(*value).ok());
        Ok(Self {
            config,
            architecture_version: checkpoint
                .get("version")
                .and_then(Value::as_str)
                .map(str::to_owned),
            sample_rate_label: checkpoint
                .get("sr")
                .and_then(Value::as_str)
                .map(str::to_owned),
            pitch_guidance: checkpoint.get("f0").map(truthy).unwrap_or(false),
            training_info: checkpoint
                .get("info")
                .and_then(Value::as_str)
                .map(str::to_owned),
            phone_feature_channels,
        })
    }

    pub fn validate(&self, checkpoint: &Checkpoint) -> ValidationReport {
        let mut report = ValidationReport::default();
        let config = &self.config;
        if config.upsample_rates.len() != config.upsample_kernel_sizes.len() {
            report
                .errors
                .push("upsample rate and kernel counts differ".into());
        }
        if config.resblock_kernel_sizes.len() != config.resblock_dilation_sizes.len() {
            report
                .errors
                .push("resblock kernel and dilation counts differ".into());
        }
        if config.sample_rate == 0 {
            report.errors.push("sample rate is zero".into());
        }
        if let Some(tensor) = checkpoint.tensor("enc_p.emb_phone.weight") {
            let expected_output = u64::from(config.hidden_channels);
            if tensor.shape.first() != Some(&expected_output) {
                report.errors.push(format!(
                    "enc_p.emb_phone.weight output is {:?}; expected {expected_output}",
                    tensor.shape.first()
                ));
            }
        } else {
            report
                .errors
                .push("enc_p.emb_phone.weight is missing".into());
        }
        if let Some(tensor) = checkpoint.tensor("emb_g.weight") {
            let expected = [
                u64::from(config.speaker_count),
                u64::from(config.speaker_embedding_channels),
            ];
            if tensor.shape.as_slice() != expected {
                report.errors.push(format!(
                    "emb_g.weight shape is {:?}; expected {:?}",
                    tensor.shape, expected
                ));
            }
        } else if config.speaker_count > 0 {
            report.warnings.push("emb_g.weight is missing".into());
        }
        if self.pitch_guidance && checkpoint.tensor("enc_p.emb_pitch.weight").is_none() {
            report
                .errors
                .push("pitch guidance is enabled but enc_p.emb_pitch.weight is missing".into());
        }
        if let Some(label) = &self.sample_rate_label {
            if let Some(rate) = parse_rate_label(label) {
                if rate != config.sample_rate {
                    report.warnings.push(format!(
                        "sample-rate label {label} differs from config value {}",
                        config.sample_rate
                    ));
                }
            }
        }
        if let (Some(version), Some(channels)) = (
            self.architecture_version.as_deref(),
            self.phone_feature_channels,
        ) {
            let expected = match version {
                "v1" => Some(256),
                "v2" => Some(768),
                _ => None,
            };
            if let Some(expected) = expected {
                if channels != expected {
                    report.warnings.push(format!(
                        "architecture {version} usually uses {expected} phone channels; found {channels}"
                    ));
                }
            }
        }
        report
    }

    pub fn validate_index_dimension(&self, dimension: usize) -> Result<()> {
        let expected = self
            .phone_feature_channels
            .ok_or_else(|| Error::InvalidArchive("phone feature dimension is unavailable".into()))?
            as usize;
        if dimension != expected {
            return Err(Error::DimensionMismatch {
                expected,
                found: dimension,
            });
        }
        Ok(())
    }
}

fn uint(value: &Value, field: &str) -> Result<u32> {
    let integer = value
        .as_i64()
        .ok_or_else(|| Error::InvalidArchive(format!("{field} is not an integer")))?;
    u32::try_from(integer).map_err(|_| Error::InvalidArchive(format!("{field} is out of range")))
}

fn number(value: &Value, field: &str) -> Result<f64> {
    match value {
        Value::Int(value) => Ok(*value as f64),
        Value::Float(value) => Ok(*value),
        _ => Err(Error::InvalidArchive(format!("{field} is not numeric"))),
    }
}

fn string(value: &Value, field: &str) -> Result<String> {
    value
        .as_str()
        .map(str::to_owned)
        .ok_or_else(|| Error::InvalidArchive(format!("{field} is not a string")))
}

fn uint_list(value: &Value, field: &str) -> Result<Vec<u32>> {
    let values = value
        .as_list()
        .ok_or_else(|| Error::InvalidArchive(format!("{field} is not a list")))?;
    values.iter().map(|value| uint(value, field)).collect()
}

fn nested_uint_list(value: &Value, field: &str) -> Result<Vec<Vec<u32>>> {
    let values = value
        .as_list()
        .ok_or_else(|| Error::InvalidArchive(format!("{field} is not a list")))?;
    values.iter().map(|value| uint_list(value, field)).collect()
}

fn truthy(value: &Value) -> bool {
    match value {
        Value::Bool(value) => *value,
        Value::Int(value) => *value != 0,
        _ => false,
    }
}

fn parse_rate_label(label: &str) -> Option<u32> {
    let label = label.trim();
    if let Some(value) = label.strip_suffix(['k', 'K']) {
        return value.parse::<u32>().ok()?.checked_mul(1000);
    }
    label.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::parse_rate_label;

    #[test]
    fn parses_sample_rate_labels() {
        assert_eq!(parse_rate_label("40k"), Some(40_000));
        assert_eq!(parse_rate_label("48000"), Some(48_000));
        assert_eq!(parse_rate_label("unknown"), None);
    }
}
