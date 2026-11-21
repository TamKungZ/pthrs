use crate::{Error, Result, Value};

use super::parser::Parser;

const MAX_CONTAINER_ITEMS: usize = 2_000_000;
const MAX_MEMO_ITEMS: usize = 2_000_000;
const MAX_DEPTH: usize = 128;
const MAX_DUPLICATED_WEIGHT: u64 = 8_000_000;

#[derive(Clone, Debug)]
pub(super) enum StackItem {
    Mark,
    Value(Value),
}

impl Parser<'_> {
    pub(super) fn push(&mut self, value: Value) -> Result<()> {
        if matches!(
            value,
            Value::List(_)
                | Value::Tuple(_)
                | Value::Map(_)
                | Value::Persistent(_)
                | Value::Reduce { .. }
        ) {
            self.measure(&value, 0)?;
        }
        self.push_item(StackItem::Value(value))
    }

    pub(super) fn push_mark(&mut self) -> Result<()> {
        self.push_item(StackItem::Mark)
    }

    pub(super) fn push_item(&mut self, item: StackItem) -> Result<()> {
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

    pub(super) fn pop_value(&mut self, offset: usize) -> Result<Value> {
        match self.stack.pop() {
            Some(StackItem::Value(value)) => Ok(value),
            Some(StackItem::Mark) => Err(Error::pickle(offset, "expected value, found MARK")),
            None => Err(Error::pickle(offset, "pickle stack underflow")),
        }
    }

    pub(super) fn pop_mark_values(&mut self, offset: usize) -> Result<Vec<Value>> {
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

    pub(super) fn last_value_mut(&mut self, offset: usize) -> Result<&mut Value> {
        match self.stack.last_mut() {
            Some(StackItem::Value(value)) => Ok(value),
            Some(StackItem::Mark) => Err(Error::pickle(offset, "expected container, found MARK")),
            None => Err(Error::pickle(offset, "pickle stack underflow")),
        }
    }

    pub(super) fn duplicate_top(&mut self, offset: usize) -> Result<()> {
        let item = match self.stack.last() {
            Some(StackItem::Value(value)) => {
                let weight = self.measure(value, 0)?;
                self.charge_duplicate(weight)?;
                match self.stack.last() {
                    Some(item) => item.clone(),
                    None => return Err(Error::pickle(offset, "DUP on empty stack")),
                }
            }
            Some(StackItem::Mark) => StackItem::Mark,
            None => return Err(Error::pickle(offset, "DUP on empty stack")),
        };
        self.push_item(item)
    }

    pub(super) fn memo_put(&mut self, index: usize, offset: usize) -> Result<()> {
        if index >= MAX_MEMO_ITEMS {
            return Err(Error::LimitExceeded {
                what: "pickle memo index",
                value: index as u64,
                limit: MAX_MEMO_ITEMS as u64,
            });
        }
        let weight = match self.stack.last() {
            Some(StackItem::Value(value)) => self.measure(value, 0)?,
            _ => return Err(Error::pickle(offset, "memo write requires a value")),
        };
        self.charge_duplicate(weight)?;
        let Some(StackItem::Value(value)) = self.stack.last() else {
            return Err(Error::pickle(offset, "memo write requires a value"));
        };
        let value = value.clone();
        if self.memo.len() <= index {
            self.memo.resize(index + 1, None);
        }
        self.memo[index] = Some(value);
        Ok(())
    }

    pub(super) fn memo_get(&mut self, index: usize, offset: usize) -> Result<()> {
        let missing = || Error::pickle(offset, format!("memo entry {index} does not exist"));
        let weight = match self.memo.get(index) {
            Some(Some(value)) => self.measure(value, 0)?,
            _ => return Err(missing()),
        };
        self.charge_duplicate(weight)?;
        let value = self
            .memo
            .get(index)
            .and_then(Clone::clone)
            .ok_or_else(missing)?;
        self.push(value)
    }

    /// Checks that a value added to the container on top of the stack keeps it
    /// within the nesting limit.
    pub(super) fn check_child(&self, child: &Value) -> Result<()> {
        self.measure(child, 1).map(|_| ())
    }

    fn charge_duplicate(&mut self, weight: u64) -> Result<()> {
        self.duplicated = self.duplicated.saturating_add(weight);
        if self.duplicated > MAX_DUPLICATED_WEIGHT {
            return Err(Error::LimitExceeded {
                what: "duplicated pickle data",
                value: self.duplicated,
                limit: MAX_DUPLICATED_WEIGHT,
            });
        }
        Ok(())
    }

    fn measure(&self, value: &Value, depth: usize) -> Result<u64> {
        let mut weight = 0;
        if measure(value, depth, &mut weight) {
            Ok(weight)
        } else {
            Err(Error::LimitExceeded {
                what: "pickle nesting depth",
                value: MAX_DEPTH as u64 + 1,
                limit: MAX_DEPTH as u64,
            })
        }
    }
}

fn measure(value: &Value, depth: usize, weight: &mut u64) -> bool {
    if depth > MAX_DEPTH {
        return false;
    }
    *weight += 1;
    match value {
        Value::String(text) => *weight += text.len() as u64 / 16,
        Value::Bytes(bytes) => *weight += bytes.len() as u64 / 16,
        Value::List(values) | Value::Tuple(values) => {
            return values.iter().all(|child| measure(child, depth + 1, weight));
        }
        Value::Map(entries) => {
            return entries.iter().all(|(key, value)| {
                measure(key, depth + 1, weight) && measure(value, depth + 1, weight)
            });
        }
        Value::Persistent(inner) => return measure(inner, depth + 1, weight),
        Value::Reduce { callable, args } => {
            return measure(callable, depth + 1, weight) && measure(args, depth + 1, weight);
        }
        _ => {}
    }
    true
}
