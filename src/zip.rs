use std::io::{Read, Seek, SeekFrom};

use crate::{Error, Result};

const EOCD_SIGNATURE: u32 = 0x0605_4b50;
const ZIP64_LOCATOR_SIGNATURE: u32 = 0x0706_4b50;
const ZIP64_EOCD_SIGNATURE: u32 = 0x0606_4b50;
const CENTRAL_SIGNATURE: u32 = 0x0201_4b50;
const LOCAL_SIGNATURE: u32 = 0x0403_4b50;
const MAX_EOCD_SEARCH: u64 = 65_535 + 22;

#[derive(Clone, Debug)]
pub(crate) struct Entry {
    pub name: String,
    pub method: u16,
    pub compressed_size: u64,
    pub uncompressed_size: u64,
    pub data_offset: u64,
}

#[derive(Debug)]
pub(crate) struct Archive {
    entries: Vec<Entry>,
}

impl Archive {
    pub fn open<R: Read + Seek>(reader: &mut R) -> Result<Self> {
        let file_len = reader.seek(SeekFrom::End(0))?;
        let search_len = file_len.min(MAX_EOCD_SEARCH) as usize;
        if search_len < 22 {
            return Err(Error::InvalidZip("file is shorter than a ZIP end record"));
        }
        reader.seek(SeekFrom::End(-(search_len as i64)))?;
        let mut tail = vec![0; search_len];
        reader.read_exact(&mut tail)?;
        let eocd_at = (0..=tail.len().saturating_sub(22))
            .rev()
            .find(|&at| {
                le_u32(&tail[at..]) == EOCD_SIGNATURE
                    && at + 22 + le_u16(&tail[at + 20..]) as usize == tail.len()
            })
            .ok_or(Error::InvalidZip(
                "end-of-central-directory record not found",
            ))?;
        if eocd_at + 22 > tail.len() {
            return Err(Error::InvalidZip(
                "truncated end-of-central-directory record",
            ));
        }
        let eocd_abs = file_len - search_len as u64 + eocd_at as u64;
        let disk = le_u16(&tail[eocd_at + 4..]);
        let central_disk = le_u16(&tail[eocd_at + 6..]);
        if disk != 0 || central_disk != 0 {
            return Err(Error::InvalidZip("multi-disk ZIP files are not supported"));
        }

        let mut count = le_u16(&tail[eocd_at + 10..]) as u64;
        let mut central_size = le_u32(&tail[eocd_at + 12..]) as u64;
        let mut central_offset = le_u32(&tail[eocd_at + 16..]) as u64;
        if count == u16::MAX as u64
            || central_size == u32::MAX as u64
            || central_offset == u32::MAX as u64
        {
            (count, central_size, central_offset) = read_zip64_eocd(reader, eocd_abs)?;
        }
        if central_offset
            .checked_add(central_size)
            .filter(|end| *end <= file_len)
            .is_none()
        {
            return Err(Error::InvalidZip("central directory lies outside the file"));
        }
        if count > 1_000_000 {
            return Err(Error::LimitExceeded {
                what: "ZIP entry count",
                value: count,
                limit: 1_000_000,
            });
        }

        reader.seek(SeekFrom::Start(central_offset))?;
        let mut entries = Vec::with_capacity(count as usize);
        for _ in 0..count {
            let mut fixed = [0u8; 46];
            reader.read_exact(&mut fixed)?;
            if le_u32(&fixed) != CENTRAL_SIGNATURE {
                return Err(Error::InvalidZip(
                    "invalid central-directory entry signature",
                ));
            }
            let flags = le_u16(&fixed[8..]);
            if flags & 1 != 0 {
                return Err(Error::InvalidZip("encrypted ZIP entries are not supported"));
            }
            let method = le_u16(&fixed[10..]);
            let compressed32 = le_u32(&fixed[20..]);
            let uncompressed32 = le_u32(&fixed[24..]);
            let name_len = le_u16(&fixed[28..]) as usize;
            let extra_len = le_u16(&fixed[30..]) as usize;
            let comment_len = le_u16(&fixed[32..]) as usize;
            let disk_start = le_u16(&fixed[34..]);
            let local32 = le_u32(&fixed[42..]);
            if disk_start != 0 && disk_start != u16::MAX {
                return Err(Error::InvalidZip("multi-disk ZIP entry is not supported"));
            }
            let mut name = vec![0; name_len];
            let mut extra = vec![0; extra_len];
            reader.read_exact(&mut name)?;
            reader.read_exact(&mut extra)?;
            reader.seek(SeekFrom::Current(comment_len as i64))?;
            let name = String::from_utf8(name)
                .map_err(|_| Error::InvalidZip("entry name is not valid UTF-8"))?;
            let (uncompressed_size, compressed_size, local_offset) =
                zip64_values(&extra, uncompressed32, compressed32, local32, disk_start)?;
            let return_to = reader.stream_position()?;
            let data_offset = local_data_offset(reader, local_offset, file_len)?;
            reader.seek(SeekFrom::Start(return_to))?;
            if data_offset
                .checked_add(compressed_size)
                .filter(|end| *end <= file_len)
                .is_none()
            {
                return Err(Error::InvalidZip("entry data lies outside the file"));
            }
            entries.push(Entry {
                name,
                method,
                compressed_size,
                uncompressed_size,
                data_offset,
            });
        }
        Ok(Self { entries })
    }

    pub fn entries(&self) -> &[Entry] {
        &self.entries
    }

    pub fn find(&self, name: &str) -> Option<&Entry> {
        self.entries.iter().find(|entry| entry.name == name)
    }

    pub fn read_all<R: Read + Seek>(
        &self,
        reader: &mut R,
        entry: &Entry,
        limit: u64,
    ) -> Result<Vec<u8>> {
        if entry.method != 0 {
            return Err(Error::UnsupportedCompression {
                method: entry.method,
                entry: entry.name.clone(),
            });
        }
        if entry.compressed_size != entry.uncompressed_size {
            return Err(Error::InvalidZip(
                "stored entry has different compressed and uncompressed sizes",
            ));
        }
        if entry.uncompressed_size > limit {
            return Err(Error::LimitExceeded {
                what: "ZIP entry",
                value: entry.uncompressed_size,
                limit,
            });
        }
        let mut bytes =
            vec![
                0;
                usize::try_from(entry.uncompressed_size).map_err(|_| Error::LimitExceeded {
                    what: "ZIP entry",
                    value: entry.uncompressed_size,
                    limit: usize::MAX as u64
                })?
            ];
        reader.seek(SeekFrom::Start(entry.data_offset))?;
        reader.read_exact(&mut bytes)?;
        Ok(bytes)
    }

    pub fn read_range<R: Read + Seek>(
        &self,
        reader: &mut R,
        entry: &Entry,
        offset: u64,
        length: u64,
    ) -> Result<Vec<u8>> {
        if entry.method != 0 {
            return Err(Error::UnsupportedCompression {
                method: entry.method,
                entry: entry.name.clone(),
            });
        }
        if entry.compressed_size != entry.uncompressed_size {
            return Err(Error::InvalidZip(
                "stored entry has different compressed and uncompressed sizes",
            ));
        }
        let end = offset
            .checked_add(length)
            .ok_or(Error::InvalidZip("entry range overflow"))?;
        if end > entry.uncompressed_size {
            return Err(Error::InvalidZip("entry range lies outside entry data"));
        }
        let mut bytes = vec![
            0;
            usize::try_from(length).map_err(|_| Error::LimitExceeded {
                what: "read length",
                value: length,
                limit: usize::MAX as u64
            })?
        ];
        reader.seek(SeekFrom::Start(entry.data_offset + offset))?;
        reader.read_exact(&mut bytes)?;
        Ok(bytes)
    }

    pub fn read_range_into<R: Read + Seek>(
        &self,
        reader: &mut R,
        entry: &Entry,
        offset: u64,
        output: &mut [u8],
    ) -> Result<()> {
        if entry.method != 0 {
            return Err(Error::UnsupportedCompression {
                method: entry.method,
                entry: entry.name.clone(),
            });
        }
        if entry.compressed_size != entry.uncompressed_size {
            return Err(Error::InvalidZip(
                "stored entry has different compressed and uncompressed sizes",
            ));
        }
        let length = output.len() as u64;
        let end = offset
            .checked_add(length)
            .ok_or(Error::InvalidZip("entry range overflow"))?;
        if end > entry.uncompressed_size {
            return Err(Error::InvalidZip("entry range lies outside entry data"));
        }
        reader.seek(SeekFrom::Start(entry.data_offset + offset))?;
        reader.read_exact(output)?;
        Ok(())
    }
}

fn local_data_offset<R: Read + Seek>(reader: &mut R, offset: u64, file_len: u64) -> Result<u64> {
    if offset
        .checked_add(30)
        .filter(|end| *end <= file_len)
        .is_none()
    {
        return Err(Error::InvalidZip(
            "local entry header lies outside the file",
        ));
    }
    reader.seek(SeekFrom::Start(offset))?;
    let mut fixed = [0u8; 30];
    reader.read_exact(&mut fixed)?;
    if le_u32(&fixed) != LOCAL_SIGNATURE {
        return Err(Error::InvalidZip("invalid local entry signature"));
    }
    let name_len = le_u16(&fixed[26..]) as u64;
    let extra_len = le_u16(&fixed[28..]) as u64;
    offset
        .checked_add(30)
        .and_then(|v| v.checked_add(name_len))
        .and_then(|v| v.checked_add(extra_len))
        .ok_or(Error::InvalidZip("local entry offset overflow"))
}

fn read_zip64_eocd<R: Read + Seek>(reader: &mut R, eocd_abs: u64) -> Result<(u64, u64, u64)> {
    if eocd_abs < 20 {
        return Err(Error::InvalidZip("ZIP64 locator is missing"));
    }
    reader.seek(SeekFrom::Start(eocd_abs - 20))?;
    let mut locator = [0u8; 20];
    reader.read_exact(&mut locator)?;
    if le_u32(&locator) != ZIP64_LOCATOR_SIGNATURE {
        return Err(Error::InvalidZip("ZIP64 locator signature is missing"));
    }
    if le_u32(&locator[4..]) != 0 || le_u32(&locator[16..]) != 1 {
        return Err(Error::InvalidZip(
            "multi-disk ZIP64 files are not supported",
        ));
    }
    let offset = le_u64(&locator[8..]);
    reader.seek(SeekFrom::Start(offset))?;
    let mut fixed = [0u8; 56];
    reader.read_exact(&mut fixed)?;
    if le_u32(&fixed) != ZIP64_EOCD_SIGNATURE {
        return Err(Error::InvalidZip(
            "ZIP64 end-of-central-directory signature is missing",
        ));
    }
    if le_u32(&fixed[16..]) != 0 || le_u32(&fixed[20..]) != 0 {
        return Err(Error::InvalidZip(
            "multi-disk ZIP64 files are not supported",
        ));
    }
    Ok((
        le_u64(&fixed[32..]),
        le_u64(&fixed[40..]),
        le_u64(&fixed[48..]),
    ))
}

fn zip64_values(
    extra: &[u8],
    u32_size: u32,
    c32_size: u32,
    local32: u32,
    disk16: u16,
) -> Result<(u64, u64, u64)> {
    let mut uncompressed = u32_size as u64;
    let mut compressed = c32_size as u64;
    let mut local = local32 as u64;
    let needs =
        u32_size == u32::MAX || c32_size == u32::MAX || local32 == u32::MAX || disk16 == u16::MAX;
    if !needs {
        return Ok((uncompressed, compressed, local));
    }
    let mut cursor = 0usize;
    while cursor + 4 <= extra.len() {
        let id = le_u16(&extra[cursor..]);
        let size = le_u16(&extra[cursor + 2..]) as usize;
        cursor += 4;
        if cursor + size > extra.len() {
            return Err(Error::InvalidZip("truncated ZIP extra field"));
        }
        if id == 0x0001 {
            let data = &extra[cursor..cursor + size];
            let mut at = 0usize;
            let mut next_u64 = || -> Result<u64> {
                if at + 8 > data.len() {
                    return Err(Error::InvalidZip("truncated ZIP64 extra field"));
                }
                let value = le_u64(&data[at..]);
                at += 8;
                Ok(value)
            };
            if u32_size == u32::MAX {
                uncompressed = next_u64()?;
            }
            if c32_size == u32::MAX {
                compressed = next_u64()?;
            }
            if local32 == u32::MAX {
                local = next_u64()?;
            }
            return Ok((uncompressed, compressed, local));
        }
        cursor += size;
    }
    Err(Error::InvalidZip("required ZIP64 extra field is missing"))
}

fn le_u16(bytes: &[u8]) -> u16 {
    u16::from_le_bytes([bytes[0], bytes[1]])
}
fn le_u32(bytes: &[u8]) -> u32 {
    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}
fn le_u64(bytes: &[u8]) -> u64 {
    u64::from_le_bytes(bytes[..8].try_into().expect("eight bytes"))
}
