use crate::{Error, Result};

const MAX_BITS: usize = 15;
const LENGTH_BASE: [u16; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131,
    163, 195, 227, 258,
];
const LENGTH_EXTRA: [u8; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];
const DISTANCE_BASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];
const DISTANCE_EXTRA: [u8; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13,
];
const CODE_LENGTH_ORDER: [usize; 19] = [
    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
];

/// The largest ratio DEFLATE can reach, used to reject sizes no stream of the
/// given length could produce before allocating the output.
const MAX_RATIO: u64 = 1032;

fn invalid(message: &'static str) -> Error {
    Error::InvalidZip(message)
}

/// Decodes a raw DEFLATE stream that must expand to exactly `expected` bytes.
pub(crate) fn inflate(input: &[u8], expected: usize) -> Result<Vec<u8>> {
    if expected as u64 > (input.len() as u64 + 1).saturating_mul(MAX_RATIO) {
        return Err(invalid(
            "entry size is impossible for its compressed length",
        ));
    }
    let mut output = Vec::new();
    output
        .try_reserve_exact(expected)
        .map_err(|_| Error::Allocation {
            bytes: expected as u64,
        })?;
    let mut state = State {
        input,
        position: 0,
        bit_buffer: 0,
        bit_count: 0,
        output,
        limit: expected,
    };
    loop {
        let last = state.bits(1)?;
        match state.bits(2)? {
            0 => state.stored()?,
            1 => {
                let (literals, distances) = fixed_tables();
                state.codes(&literals, &distances)?;
            }
            2 => {
                let (literals, distances) = state.dynamic_tables()?;
                state.codes(&literals, &distances)?;
            }
            _ => return Err(invalid("invalid DEFLATE block type")),
        }
        if last == 1 {
            break;
        }
    }
    if state.output.len() != expected {
        return Err(invalid("DEFLATE stream is shorter than the entry size"));
    }
    Ok(state.output)
}

struct Huffman {
    count: [u16; MAX_BITS + 1],
    symbol: Vec<u16>,
}

impl Huffman {
    fn new(lengths: &[u8]) -> Result<Self> {
        let mut count = [0u16; MAX_BITS + 1];
        for &length in lengths {
            count[length as usize] += 1;
        }
        let mut left: i32 = 1;
        for &entries in &count[1..] {
            left = (left << 1) - i32::from(entries);
            if left < 0 {
                return Err(invalid("over-subscribed DEFLATE code"));
            }
        }
        let mut offsets = [0u16; MAX_BITS + 2];
        for length in 1..=MAX_BITS {
            offsets[length + 1] = offsets[length] + count[length];
        }
        let mut symbol = vec![0u16; lengths.len()];
        for (index, &length) in lengths.iter().enumerate() {
            if length != 0 {
                symbol[offsets[length as usize] as usize] = index as u16;
                offsets[length as usize] += 1;
            }
        }
        Ok(Self { count, symbol })
    }
}

fn fixed_tables() -> (Huffman, Huffman) {
    let mut lengths = [8u8; 288];
    lengths[144..256].fill(9);
    lengths[256..280].fill(7);
    let literals = Huffman::new(&lengths).expect("fixed literal code is valid");
    let distances = Huffman::new(&[5u8; 30]).expect("fixed distance code is valid");
    (literals, distances)
}

struct State<'a> {
    input: &'a [u8],
    position: usize,
    bit_buffer: u32,
    bit_count: u32,
    output: Vec<u8>,
    limit: usize,
}

impl State<'_> {
    fn bits(&mut self, need: u32) -> Result<u32> {
        while self.bit_count < need {
            let byte = *self
                .input
                .get(self.position)
                .ok_or_else(|| invalid("truncated DEFLATE stream"))?;
            self.position += 1;
            self.bit_buffer |= u32::from(byte) << self.bit_count;
            self.bit_count += 8;
        }
        let value = self.bit_buffer & ((1u32 << need) - 1);
        self.bit_buffer >>= need;
        self.bit_count -= need;
        Ok(value)
    }

    fn decode(&mut self, table: &Huffman) -> Result<usize> {
        let (mut code, mut first, mut index) = (0i32, 0i32, 0i32);
        for length in 1..=MAX_BITS {
            code |= self.bits(1)? as i32;
            let count = i32::from(table.count[length]);
            if code - count < first {
                return Ok(table.symbol[(index + (code - first)) as usize] as usize);
            }
            index += count;
            first = (first + count) << 1;
            code <<= 1;
        }
        Err(invalid("invalid DEFLATE code"))
    }

    fn stored(&mut self) -> Result<()> {
        self.bit_buffer = 0;
        self.bit_count = 0;
        let header = self
            .input
            .get(self.position..self.position + 4)
            .ok_or_else(|| invalid("truncated DEFLATE stored block"))?;
        let length = usize::from(u16::from_le_bytes([header[0], header[1]]));
        let complement = u16::from_le_bytes([header[2], header[3]]);
        if length as u16 != !complement {
            return Err(invalid("corrupt DEFLATE stored block"));
        }
        self.position += 4;
        let data = self
            .input
            .get(self.position..self.position + length)
            .ok_or_else(|| invalid("truncated DEFLATE stored block"))?;
        self.position += length;
        if self.output.len() + length > self.limit {
            return Err(invalid("DEFLATE stream is longer than the entry size"));
        }
        self.output.extend_from_slice(data);
        Ok(())
    }

    fn dynamic_tables(&mut self) -> Result<(Huffman, Huffman)> {
        let literal_count = self.bits(5)? as usize + 257;
        let distance_count = self.bits(5)? as usize + 1;
        let code_count = self.bits(4)? as usize + 4;
        if literal_count > 286 || distance_count > 30 {
            return Err(invalid("too many DEFLATE length or distance codes"));
        }
        let mut code_lengths = [0u8; 19];
        for &slot in &CODE_LENGTH_ORDER[..code_count] {
            code_lengths[slot] = self.bits(3)? as u8;
        }
        let code_table = Huffman::new(&code_lengths)?;
        let total = literal_count + distance_count;
        let mut lengths = [0u8; 286 + 30];
        let mut index = 0;
        while index < total {
            let symbol = self.decode(&code_table)?;
            if symbol < 16 {
                lengths[index] = symbol as u8;
                index += 1;
                continue;
            }
            let (value, repeat) = match symbol {
                16 => {
                    if index == 0 {
                        return Err(invalid("DEFLATE repeat without a previous length"));
                    }
                    (lengths[index - 1], 3 + self.bits(2)? as usize)
                }
                17 => (0, 3 + self.bits(3)? as usize),
                _ => (0, 11 + self.bits(7)? as usize),
            };
            if index + repeat > total {
                return Err(invalid("DEFLATE code lengths overflow"));
            }
            lengths[index..index + repeat].fill(value);
            index += repeat;
        }
        if lengths[256] == 0 {
            return Err(invalid("DEFLATE block has no end-of-block code"));
        }
        let literals = Huffman::new(&lengths[..literal_count])?;
        let distances = Huffman::new(&lengths[literal_count..total])?;
        Ok((literals, distances))
    }

    fn codes(&mut self, literals: &Huffman, distances: &Huffman) -> Result<()> {
        loop {
            let symbol = self.decode(literals)?;
            if symbol < 256 {
                if self.output.len() >= self.limit {
                    return Err(invalid("DEFLATE stream is longer than the entry size"));
                }
                self.output.push(symbol as u8);
                continue;
            }
            if symbol == 256 {
                return Ok(());
            }
            let symbol = symbol - 257;
            if symbol >= LENGTH_BASE.len() {
                return Err(invalid("invalid DEFLATE length code"));
            }
            let length = usize::from(LENGTH_BASE[symbol])
                + self.bits(u32::from(LENGTH_EXTRA[symbol]))? as usize;
            let symbol = self.decode(distances)?;
            if symbol >= DISTANCE_BASE.len() {
                return Err(invalid("invalid DEFLATE distance code"));
            }
            let distance = usize::from(DISTANCE_BASE[symbol])
                + self.bits(u32::from(DISTANCE_EXTRA[symbol]))? as usize;
            if distance > self.output.len() {
                return Err(invalid("DEFLATE distance reaches before the output"));
            }
            if self.output.len() + length > self.limit {
                return Err(invalid("DEFLATE stream is longer than the entry size"));
            }
            for _ in 0..length {
                let byte = self.output[self.output.len() - distance];
                self.output.push(byte);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unhex(text: &str) -> Vec<u8> {
        (0..text.len())
            .step_by(2)
            .map(|at| u8::from_str_radix(&text[at..at + 2], 16).unwrap())
            .collect()
    }

    #[test]
    fn decodes_stored_block() {
        let stream = [0x01, 3, 0, 0xfc, 0xff, b'a', b'b', b'c'];
        assert_eq!(inflate(&stream, 3).unwrap(), b"abc");
    }

    #[test]
    fn decodes_fixed_huffman_block() {
        let stream = unhex("4b4c4a4e444500");
        assert_eq!(inflate(&stream, 18).unwrap(), b"abcabcabcabcabcabc");
    }

    #[test]
    fn decodes_dynamic_huffman_block() {
        let stream = unhex(concat!(
            "7dd43b0ec2301045d1ad640108cddf76c14a10054544680091486c9f029a9948afbed579f2789b1f",
            "ebf37de4e367bedf966d5a97eb6b3e9df530715ca6ed57a5d448554b1da95aa92c297bcd9e72d4dc",
            "53a6928553ae2a31c892065d4a10a60a611a10a603c24c20cc1cc2ac43983384b9419837080b82b0",
            "50088bc00f71405813086b0e61ad43586708eb0661bd41d8207c620a6123af566123afb6bb31cab3",
            "551953deadd298f270d5c69c97ab38e63cddee07e1bcddee0b91ff785f"
        ));
        let expected: String = (1..=40)
            .map(|i| format!("tensor.{}.weight shape=[{}, 16] ", i % 7, i * 3))
            .collect();
        assert_eq!(
            inflate(&stream, expected.len()).unwrap(),
            expected.as_bytes()
        );
    }

    #[test]
    fn rejects_wrong_sizes_and_corrupt_streams() {
        let stream = unhex("4b4c4a4e444500");
        assert!(inflate(&stream, 17).is_err());
        assert!(inflate(&stream, 19).is_err());
        assert!(inflate(&stream[..3], 18).is_err());
        assert!(inflate(&stream, 1 << 40).is_err());
        assert!(inflate(&[0x07, 0, 0, 0], 1).is_err());
        assert!(inflate(&[0x01, 3, 0, 0, 0, b'a', b'b', b'c'], 3).is_err());
    }
}
