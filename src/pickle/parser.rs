use crate::{Error, Result, Value};

use super::{
    literal::{
        expect_string, line_string, map_insert, pairs, parse_i64, parse_quoted, parse_usize,
        signed_le, type_error,
    },
    reduce::{persistent, reduce},
};

const MAX_CONTAINER_ITEMS: usize = 2_000_000;
const MAX_MEMO_ITEMS: usize = 2_000_000;
const MAX_INLINE_BYTES: u64 = 256 * 1024 * 1024;

#[derive(Clone, Debug)]
enum StackItem {
    Mark,
    Value(Value),
}

pub(super) struct Parser<'a> {
    bytes: &'a [u8],
    at: usize,
    protocol: u8,
    stack: Vec<StackItem>,
    memo: Vec<Option<Value>>,
}

impl<'a> Parser<'a> {
    pub(super) fn new(bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            at: 0,
            protocol: 0,
            stack: Vec::new(),
            memo: Vec::new(),
        }
    }

    pub(super) fn run(mut self) -> Result<(u8, Value)> {
        loop {
            let opcode_at = self.at;
            let opcode = self.byte()?;
            match opcode {
                b'.' => {
                    let value = self.pop_value(opcode_at)?;
                    if self
                        .stack
                        .iter()
                        .any(|item| matches!(item, StackItem::Mark))
                    {
                        return Err(Error::pickle(opcode_at, "unclosed MARK at STOP"));
                    }
                    return Ok((self.protocol, value));
                }
                0x80 => self.protocol = self.byte()?, // PROTO
                0x95 => {
                    self.take_u64()?;
                } // FRAME; frame length is advisory here
                b'(' => self.push_mark()?,
                b'0' => {
                    self.stack
                        .pop()
                        .ok_or_else(|| Error::pickle(opcode_at, "POP on empty stack"))?;
                }
                b'1' => {
                    self.pop_mark_values(opcode_at)?;
                }
                b'2' => {
                    let value = self
                        .stack
                        .last()
                        .cloned()
                        .ok_or_else(|| Error::pickle(opcode_at, "DUP on empty stack"))?;
                    self.push_item(value)?;
                }
                b'N' => self.push(Value::None)?,
                0x88 => self.push(Value::Bool(true))?,
                0x89 => self.push(Value::Bool(false))?,
                b'I' => {
                    let text = self.line()?;
                    let value = match text {
                        b"00" => Value::Bool(false),
                        b"01" => Value::Bool(true),
                        _ => Value::Int(parse_i64(text, opcode_at)?),
                    };
                    self.push(value)?;
                }
                b'J' => {
                    let value = self.take_i32()? as i64;
                    self.push(Value::Int(value))?;
                }
                b'K' => {
                    let value = self.byte()? as i64;
                    self.push(Value::Int(value))?;
                }
                b'M' => {
                    let value = self.take_u16()? as i64;
                    self.push(Value::Int(value))?;
                }
                b'L' => {
                    let mut text = self.line()?;
                    if text.last() == Some(&b'L') {
                        text = &text[..text.len() - 1];
                    }
                    let value = parse_i64(text, opcode_at)?;
                    self.push(Value::Int(value))?;
                }
                0x8a => {
                    let len = self.byte()? as usize;
                    let value = signed_le(self.take(len)?, opcode_at)?;
                    self.push(Value::Int(value))?;
                }
                0x8b => {
                    let len = self.take_u32()? as u64;
                    let value = signed_le(self.take_len(len)?, opcode_at)?;
                    self.push(Value::Int(value))?;
                }
                b'F' => {
                    let text = self.line()?;
                    let value = std::str::from_utf8(text)
                        .map_err(|_| Error::pickle(opcode_at, "invalid FLOAT"))?
                        .parse::<f64>()
                        .map_err(|_| Error::pickle(opcode_at, "invalid FLOAT"))?;
                    self.push(Value::Float(value))?;
                }
                b'G' => {
                    let bytes: [u8; 8] = self.take(8)?.try_into().expect("eight bytes");
                    self.push(Value::Float(f64::from_be_bytes(bytes)))?;
                }
                b'X' => {
                    let len = self.take_u32()? as u64;
                    let string = self.utf8(len, opcode_at)?;
                    self.push(Value::String(string))?;
                }
                0x8c => {
                    let len = self.byte()? as u64;
                    let string = self.utf8(len, opcode_at)?;
                    self.push(Value::String(string))?;
                }
                0x8d => {
                    let len = self.take_u64()?;
                    let string = self.utf8(len, opcode_at)?;
                    self.push(Value::String(string))?;
                }
                b'V' => {
                    let string = String::from_utf8_lossy(self.line()?).into_owned();
                    self.push(Value::String(string))?;
                }
                b'S' => {
                    let string = parse_quoted(self.line()?, opcode_at)?;
                    self.push(Value::String(string))?;
                }
                b'T' => {
                    let len = self.take_u32()? as u64;
                    let value = self.take_len(len)?.to_vec();
                    self.push(Value::Bytes(value))?;
                }
                b'U' => {
                    let len = self.byte()? as usize;
                    let value = self.take(len)?.to_vec();
                    self.push(Value::Bytes(value))?;
                }
                b'B' => {
                    let len = self.take_u32()? as u64;
                    let value = self.take_len(len)?.to_vec();
                    self.push(Value::Bytes(value))?;
                }
                b'C' => {
                    let len = self.byte()? as usize;
                    let value = self.take(len)?.to_vec();
                    self.push(Value::Bytes(value))?;
                }
                0x8e | 0x96 => {
                    let len = self.take_u64()?;
                    let value = self.take_len(len)?.to_vec();
                    self.push(Value::Bytes(value))?;
                }
                b']' => self.push(Value::List(Vec::new()))?,
                b'l' => {
                    let values = self.pop_mark_values(opcode_at)?;
                    self.push(Value::List(values))?;
                }
                b'a' => {
                    let value = self.pop_value(opcode_at)?;
                    match self.last_value_mut(opcode_at)? {
                        Value::List(values) => values.push(value),
                        other => return Err(type_error(opcode_at, "list", other.kind())),
                    }
                }
                b'e' => {
                    let values = self.pop_mark_values(opcode_at)?;
                    match self.last_value_mut(opcode_at)? {
                        Value::List(target) => target.extend(values),
                        other => return Err(type_error(opcode_at, "list", other.kind())),
                    }
                }
                b')' => self.push(Value::Tuple(Vec::new()))?,
                b't' => {
                    let values = self.pop_mark_values(opcode_at)?;
                    self.push(Value::Tuple(values))?;
                }
                0x85 => {
                    let a = self.pop_value(opcode_at)?;
                    self.push(Value::Tuple(vec![a]))?;
                }
                0x86 => {
                    let b = self.pop_value(opcode_at)?;
                    let a = self.pop_value(opcode_at)?;
                    self.push(Value::Tuple(vec![a, b]))?;
                }
                0x87 => {
                    let c = self.pop_value(opcode_at)?;
                    let b = self.pop_value(opcode_at)?;
                    let a = self.pop_value(opcode_at)?;
                    self.push(Value::Tuple(vec![a, b, c]))?;
                }
                b'}' => self.push(Value::Map(Vec::new()))?,
                b'd' => {
                    let values = self.pop_mark_values(opcode_at)?;
                    let entries = pairs(values, opcode_at)?;
                    self.push(Value::Map(entries))?;
                }
                b's' => {
                    let value = self.pop_value(opcode_at)?;
                    let key = self.pop_value(opcode_at)?;
                    match self.last_value_mut(opcode_at)? {
                        Value::Map(entries) => map_insert(entries, key, value),
                        other => return Err(type_error(opcode_at, "map", other.kind())),
                    }
                }
                b'u' => {
                    let entries = pairs(self.pop_mark_values(opcode_at)?, opcode_at)?;
                    match self.last_value_mut(opcode_at)? {
                        Value::Map(target) => {
                            for (key, value) in entries {
                                map_insert(target, key, value);
                            }
                        }
                        other => return Err(type_error(opcode_at, "map", other.kind())),
                    }
                }
                0x8f => self.push(Value::List(Vec::new()))?, // EMPTY_SET, represented as a list
                0x90 => {
                    let values = self.pop_mark_values(opcode_at)?;
                    match self.last_value_mut(opcode_at)? {
                        Value::List(target) => target.extend(values),
                        other => return Err(type_error(opcode_at, "set", other.kind())),
                    }
                }
                0x91 => {
                    let values = self.pop_mark_values(opcode_at)?;
                    self.push(Value::List(values))?;
                }
                b'c' => {
                    let module = line_string(self.line()?, opcode_at)?;
                    let name = line_string(self.line()?, opcode_at)?;
                    self.push(Value::Global { module, name })?;
                }
                0x93 => {
                    let name = expect_string(self.pop_value(opcode_at)?, opcode_at)?;
                    let module = expect_string(self.pop_value(opcode_at)?, opcode_at)?;
                    self.push(Value::Global { module, name })?;
                }
                b'R' => {
                    let args = self.pop_value(opcode_at)?;
                    let callable = self.pop_value(opcode_at)?;
                    let value = reduce(callable, args, opcode_at)?;
                    self.push(value)?;
                }
                b'i' => {
                    let module = line_string(self.line()?, opcode_at)?;
                    let name = line_string(self.line()?, opcode_at)?;
                    let args = self.pop_mark_values(opcode_at)?;
                    self.push(Value::Reduce {
                        callable: Box::new(Value::Global { module, name }),
                        args: Box::new(Value::Tuple(args)),
                    })?;
                }
                b'o' => {
                    let mut values = self.pop_mark_values(opcode_at)?.into_iter();
                    let callable = values
                        .next()
                        .ok_or_else(|| Error::pickle(opcode_at, "OBJ without a class"))?;
                    self.push(Value::Reduce {
                        callable: Box::new(callable),
                        args: Box::new(Value::Tuple(values.collect())),
                    })?;
                }
                0x81 => {
                    // NEWOBJ is inert and equivalent to a reduction for our data model
                    let args = self.pop_value(opcode_at)?;
                    let callable = self.pop_value(opcode_at)?;
                    self.push(Value::Reduce {
                        callable: Box::new(callable),
                        args: Box::new(args),
                    })?;
                }
                0x92 => {
                    let kwargs = self.pop_value(opcode_at)?;
                    let args = self.pop_value(opcode_at)?;
                    let callable = self.pop_value(opcode_at)?;
                    self.push(Value::Reduce {
                        callable: Box::new(callable),
                        args: Box::new(Value::Tuple(vec![args, kwargs])),
                    })?;
                }
                b'b' => {
                    let _state = self.pop_value(opcode_at)?;
                } // BUILD state is deliberately not executed
                b'Q' => {
                    let id = self.pop_value(opcode_at)?;
                    let value = persistent(id, opcode_at)?;
                    self.push(value)?;
                }
                b'P' => {
                    let id = Value::String(line_string(self.line()?, opcode_at)?);
                    let value = persistent(id, opcode_at)?;
                    self.push(value)?;
                }
                b'q' => {
                    let index = self.byte()? as usize;
                    self.memo_put(index, opcode_at)?;
                }
                b'r' => {
                    let index = self.take_u32()? as usize;
                    self.memo_put(index, opcode_at)?;
                }
                b'p' => {
                    let index = parse_usize(self.line()?, opcode_at)?;
                    self.memo_put(index, opcode_at)?;
                }
                b'h' => {
                    let index = self.byte()? as usize;
                    self.memo_get(index, opcode_at)?;
                }
                b'j' => {
                    let index = self.take_u32()? as usize;
                    self.memo_get(index, opcode_at)?;
                }
                b'g' => {
                    let index = parse_usize(self.line()?, opcode_at)?;
                    self.memo_get(index, opcode_at)?;
                }
                0x94 => {
                    let index = self.memo.len();
                    self.memo_put(index, opcode_at)?;
                }
                _ => {
                    return Err(Error::UnsupportedPickle {
                        offset: opcode_at,
                        opcode,
                    })
                }
            }
        }
    }

    fn byte(&mut self) -> Result<u8> {
        let value = *self
            .bytes
            .get(self.at)
            .ok_or_else(|| Error::pickle(self.at, "unexpected end of pickle"))?;
        self.at += 1;
        Ok(value)
    }
    fn take(&mut self, length: usize) -> Result<&[u8]> {
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
    fn take_len(&mut self, length: u64) -> Result<&[u8]> {
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
    fn take_u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(
            self.take(2)?.try_into().expect("two bytes"),
        ))
    }
    fn take_u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(
            self.take(4)?.try_into().expect("four bytes"),
        ))
    }
    fn take_i32(&mut self) -> Result<i32> {
        Ok(i32::from_le_bytes(
            self.take(4)?.try_into().expect("four bytes"),
        ))
    }
    fn take_u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(
            self.take(8)?.try_into().expect("eight bytes"),
        ))
    }
    fn line(&mut self) -> Result<&[u8]> {
        let rest = &self.bytes[self.at..];
        let length = rest
            .iter()
            .position(|byte| *byte == b'\n')
            .ok_or_else(|| Error::pickle(self.at, "unterminated line"))?;
        let line = &rest[..length];
        self.at += length + 1;
        Ok(line)
    }
    fn utf8(&mut self, length: u64, offset: usize) -> Result<String> {
        String::from_utf8(self.take_len(length)?.to_vec())
            .map_err(|_| Error::pickle(offset, "invalid UTF-8 string"))
    }
    fn push(&mut self, value: Value) -> Result<()> {
        self.push_item(StackItem::Value(value))
    }
    fn push_mark(&mut self) -> Result<()> {
        self.push_item(StackItem::Mark)
    }
    fn push_item(&mut self, item: StackItem) -> Result<()> {
        if self.stack.len() >= MAX_CONTAINER_ITEMS {
            return Err(Error::LimitExceeded {
                what: "pickle stack",
                value: self.stack.len() as u64,
                limit: MAX_CONTAINER_ITEMS as u64,
            });
        }
        self.stack.push(item);
        Ok(())
    }
    fn pop_value(&mut self, offset: usize) -> Result<Value> {
        match self.stack.pop() {
            Some(StackItem::Value(value)) => Ok(value),
            Some(StackItem::Mark) => Err(Error::pickle(offset, "expected value, found MARK")),
            None => Err(Error::pickle(offset, "pickle stack underflow")),
        }
    }
    fn pop_mark_values(&mut self, offset: usize) -> Result<Vec<Value>> {
        let mut values = Vec::new();
        loop {
            match self.stack.pop() {
                Some(StackItem::Value(value)) => values.push(value),
                Some(StackItem::Mark) => {
                    values.reverse();
                    return Ok(values);
                }
                None => return Err(Error::pickle(offset, "MARK not found")),
            }
        }
    }
    fn last_value_mut(&mut self, offset: usize) -> Result<&mut Value> {
        match self.stack.last_mut() {
            Some(StackItem::Value(value)) => Ok(value),
            Some(StackItem::Mark) => Err(Error::pickle(offset, "expected container, found MARK")),
            None => Err(Error::pickle(offset, "pickle stack underflow")),
        }
    }
    fn memo_put(&mut self, index: usize, offset: usize) -> Result<()> {
        if index >= MAX_MEMO_ITEMS {
            return Err(Error::LimitExceeded {
                what: "pickle memo index",
                value: index as u64,
                limit: MAX_MEMO_ITEMS as u64,
            });
        }
        let value = match self.stack.last() {
            Some(StackItem::Value(value)) => value.clone(),
            _ => return Err(Error::pickle(offset, "memo write requires a value")),
        };
        if self.memo.len() <= index {
            self.memo.resize(index + 1, None);
        }
        self.memo[index] = Some(value);
        Ok(())
    }
    fn memo_get(&mut self, index: usize, offset: usize) -> Result<()> {
        let value =
            self.memo.get(index).and_then(Clone::clone).ok_or_else(|| {
                Error::pickle(offset, format!("memo entry {index} does not exist"))
            })?;
        self.push(value)
    }
}
