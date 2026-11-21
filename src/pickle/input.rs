use crate::{Error, Result};

use super::parser::Parser;

const MAX_INLINE_BYTES: u64 = 256 * 1024 * 1024;

impl Parser<'_> {
    pub(super) fn byte(&mut self) -> Result<u8> {
        let value = *self
            .bytes
            .get(self.at)
            .ok_or_else(|| Error::pickle(self.at, "unexpected end of pickle"))?;
        self.at += 1;
        Ok(value)
    }
    pub(super) fn take(&mut self, length: usize) -> Result<&[u8]> {
        let end = self
            .at
            .checked_add(length)
            .ok_or_else(|| Error::pickle(self.at, "length overflow"))?;
        let bytes = self
            .bytes
            .get(self.at..end)
            .ok_or_else(|| Error::pickle(self.at, "truncated opcode payload"))?;
        self.at = end;
        Ok(bytes)
    }
    pub(super) fn take_len(&mut self, length: u64) -> Result<&[u8]> {
        if length > MAX_INLINE_BYTES {
            return Err(Error::LimitExceeded {
                what: "pickle byte string",
                value: length,
                limit: MAX_INLINE_BYTES,
            });
        }
        let length = usize::try_from(length)
            .map_err(|_| Error::pickle(self.at, "length does not fit this platform"))?;
        self.take(length)
    }
    pub(super) fn take_u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(
            self.take(2)?.try_into().expect("two bytes"),
        ))
    }
    pub(super) fn take_u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(
            self.take(4)?.try_into().expect("four bytes"),
        ))
    }
    pub(super) fn take_i32(&mut self) -> Result<i32> {
        Ok(i32::from_le_bytes(
            self.take(4)?.try_into().expect("four bytes"),
        ))
    }
    pub(super) fn take_u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(
            self.take(8)?.try_into().expect("eight bytes"),
        ))
    }
    pub(super) fn line(&mut self) -> Result<&[u8]> {
        let rest = &self.bytes[self.at..];
        let length = rest
            .iter()
            .position(|byte| *byte == b'\n')
            .ok_or_else(|| Error::pickle(self.at, "unterminated line"))?;
        let line = &rest[..length];
        self.at += length + 1;
        Ok(line)
    }
    pub(super) fn utf8(&mut self, length: u64, offset: usize) -> Result<String> {
        String::from_utf8(self.take_len(length)?.to_vec())
            .map_err(|_| Error::pickle(offset, "invalid UTF-8 string"))
    }
}
