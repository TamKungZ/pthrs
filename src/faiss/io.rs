use std::io::{Read, Seek, SeekFrom};

use crate::{Error, Result};

pub(super) fn read_f32_vec<R: Read>(reader: &mut R, len: usize) -> Result<Vec<f32>> {
    let mut bytes = vec![
        0;
        len.checked_mul(4).ok_or_else(|| Error::InvalidIndex(
            "float vector byte length overflow".into()
        ))?
    ];
    reader.read_exact(&mut bytes)?;
    Ok(bytes
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four bytes")))
        .collect())
}

pub(super) fn checked_skip<R: Seek>(reader: &mut R, bytes: u64, file_len: u64) -> Result<()> {
    let current = reader.stream_position()?;
    let target = current
        .checked_add(bytes)
        .ok_or_else(|| Error::InvalidIndex("file offset overflow".into()))?;
    if target > file_len {
        return Err(Error::InvalidIndex("truncated index data".into()));
    }
    reader.seek(SeekFrom::Start(target))?;
    Ok(())
}

pub(super) fn read_usize64<R: Read>(reader: &mut R, field: &str) -> Result<usize> {
    let value = read_u64(reader)?;
    usize::try_from(value)
        .map_err(|_| Error::InvalidIndex(format!("{field} does not fit this platform")))
}

pub(super) fn expect_magic<R: Read>(reader: &mut R, expected: [u8; 4], name: &str) -> Result<()> {
    let found = read_magic(reader)?;
    if found != expected {
        return Err(Error::InvalidIndex(format!(
            "expected {name} magic {:?}, found {:?}",
            magic_string(expected),
            magic_string(found)
        )));
    }
    Ok(())
}

pub(super) fn magic_string(magic: [u8; 4]) -> String {
    String::from_utf8_lossy(&magic).into_owned()
}
pub(super) fn read_magic<R: Read>(reader: &mut R) -> Result<[u8; 4]> {
    let mut bytes = [0; 4];
    reader.read_exact(&mut bytes)?;
    Ok(bytes)
}
pub(super) fn read_u8<R: Read>(reader: &mut R) -> Result<u8> {
    let mut bytes = [0; 1];
    reader.read_exact(&mut bytes)?;
    Ok(bytes[0])
}
pub(super) fn read_i32<R: Read>(reader: &mut R) -> Result<i32> {
    let mut bytes = [0; 4];
    reader.read_exact(&mut bytes)?;
    Ok(i32::from_le_bytes(bytes))
}
pub(super) fn read_u64<R: Read>(reader: &mut R) -> Result<u64> {
    let mut bytes = [0; 8];
    reader.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}
pub(super) fn read_i64<R: Read>(reader: &mut R) -> Result<i64> {
    let mut bytes = [0; 8];
    reader.read_exact(&mut bytes)?;
    Ok(i64::from_le_bytes(bytes))
}
pub(super) fn read_f32<R: Read>(reader: &mut R) -> Result<f32> {
    let mut bytes = [0; 4];
    reader.read_exact(&mut bytes)?;
    Ok(f32::from_le_bytes(bytes))
}
