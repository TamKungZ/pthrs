use crate::{Error, Result, Value};

pub(super) fn pairs(values: Vec<Value>, offset: usize) -> Result<Vec<(Value, Value)>> {
    if values.len() % 2 != 0 {
        return Err(Error::pickle(offset, "dictionary item list has odd length"));
    }
    let mut iterator = values.into_iter();
    Ok(std::iter::from_fn(|| Some((iterator.next()?, iterator.next()?))).collect())
}
const MAX_MAP_ENTRIES: usize = 100_000;

pub(super) fn map_insert(
    entries: &mut Vec<(Value, Value)>,
    key: Value,
    value: Value,
) -> Result<()> {
    if let Some(entry) = entries.iter_mut().find(|(candidate, _)| candidate == &key) {
        entry.1 = value;
        return Ok(());
    }
    if entries.len() >= MAX_MAP_ENTRIES {
        return Err(Error::LimitExceeded {
            what: "pickle map entries",
            value: entries.len() as u64 + 1,
            limit: MAX_MAP_ENTRIES as u64,
        });
    }
    entries.push((key, value));
    Ok(())
}
pub(super) fn expect_string(value: Value, offset: usize) -> Result<String> {
    match value {
        Value::String(value) => Ok(value),
        other => Err(type_error(offset, "string", other.kind())),
    }
}
pub(super) fn line_string(bytes: &[u8], offset: usize) -> Result<String> {
    String::from_utf8(bytes.to_vec()).map_err(|_| Error::pickle(offset, "line is not UTF-8"))
}
pub(super) fn parse_i64(bytes: &[u8], offset: usize) -> Result<i64> {
    std::str::from_utf8(bytes)
        .map_err(|_| Error::pickle(offset, "invalid integer"))?
        .parse()
        .map_err(|_| Error::pickle(offset, "integer does not fit i64"))
}
pub(super) fn parse_usize(bytes: &[u8], offset: usize) -> Result<usize> {
    std::str::from_utf8(bytes)
        .map_err(|_| Error::pickle(offset, "invalid memo index"))?
        .parse()
        .map_err(|_| Error::pickle(offset, "invalid memo index"))
}
pub(super) fn signed_le(bytes: &[u8], offset: usize) -> Result<i64> {
    if bytes.is_empty() {
        return Ok(0);
    }
    if bytes.len() > 8 {
        return Err(Error::pickle(offset, "LONG value does not fit i64"));
    }
    let negative = bytes.last().is_some_and(|value| value & 0x80 != 0);
    let mut full = [if negative { 0xff } else { 0 }; 8];
    full[..bytes.len()].copy_from_slice(bytes);
    Ok(i64::from_le_bytes(full))
}
pub(super) fn parse_quoted(bytes: &[u8], offset: usize) -> Result<String> {
    if bytes.len() < 2
        || !matches!(
            (bytes[0], bytes[bytes.len() - 1]),
            (b'\'', b'\'') | (b'"', b'"')
        )
    {
        return Err(Error::pickle(offset, "invalid quoted STRING"));
    }
    let mut out = String::new();
    let mut at = 1;
    while at + 1 < bytes.len() {
        if bytes[at] == b'\\' && at + 2 < bytes.len() {
            at += 1;
            let hex = (bytes[at] == b'x')
                .then(|| bytes.get(at + 1..at + 3))
                .flatten()
                .filter(|_| at + 3 < bytes.len())
                .and_then(|digits| std::str::from_utf8(digits).ok())
                .and_then(|digits| u8::from_str_radix(digits, 16).ok());
            if let Some(value) = hex {
                out.push(value as char);
                at += 3;
                continue;
            }
            out.push(match bytes[at] {
                b'n' => '\n',
                b'r' => '\r',
                b't' => '\t',
                b'\\' => '\\',
                b'\'' => '\'',
                b'"' => '"',
                other => other as char,
            });
        } else {
            out.push(bytes[at] as char);
        }
        at += 1;
    }
    Ok(out)
}
pub(super) fn type_error(offset: usize, expected: &str, found: &str) -> Error {
    Error::pickle(offset, format!("expected {expected}, found {found}"))
}
