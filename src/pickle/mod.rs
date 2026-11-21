mod input;
mod literal;
mod parser;
mod reduce;
mod stack;

use crate::{Result, Value};

/// Parse a pickle byte stream without importing or executing Python code.
///
/// This supports protocols 0 through 5 at the primitive/container level and
/// the PyTorch storage/tensor reductions used by common checkpoints. Unsupported
/// executable constructs are rejected or retained as inert [`Value`] nodes.
pub fn parse_pickle(bytes: &[u8]) -> Result<Value> {
    parse_pickle_with_protocol(bytes).map(|(_, value)| value)
}

pub(crate) fn parse_pickle_with_protocol(bytes: &[u8]) -> Result<(u8, Value)> {
    parse_pickle_prefix(bytes).map(|(protocol, value, _)| (protocol, value))
}

/// Parses the first pickle in `bytes` and reports how many bytes it used.
pub(crate) fn parse_pickle_prefix(bytes: &[u8]) -> Result<(u8, Value, usize)> {
    parser::Parser::new(bytes).run()
}

#[cfg(test)]
mod tests;
