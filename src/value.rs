use crate::{StorageRef, TensorMeta};

/// A safe, inert representation of a value decoded from `data.pkl`.
///
/// Python callables are never imported or executed. Unknown reductions remain
/// visible as [`Value::Reduce`] nodes for inspection.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
    Bytes(Vec<u8>),
    List(Vec<Value>),
    Tuple(Vec<Value>),
    Map(Vec<(Value, Value)>),
    Global {
        module: String,
        name: String,
    },
    Persistent(Box<Value>),
    Storage(StorageRef),
    Tensor(TensorMeta),
    Reduce {
        callable: Box<Value>,
        args: Box<Value>,
    },
}

impl Value {
    pub fn kind(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Bool(_) => "bool",
            Self::Int(_) => "integer",
            Self::Float(_) => "float",
            Self::String(_) => "string",
            Self::Bytes(_) => "bytes",
            Self::List(_) => "list",
            Self::Tuple(_) => "tuple",
            Self::Map(_) => "map",
            Self::Global { .. } => "global",
            Self::Persistent(_) => "persistent id",
            Self::Storage(_) => "storage",
            Self::Tensor(_) => "tensor",
            Self::Reduce { .. } => "reduction",
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(value) => Some(value),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Self::Int(value) => Some(*value),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(value) => Some(*value),
            _ => None,
        }
    }

    pub fn as_list(&self) -> Option<&[Value]> {
        match self {
            Self::List(value) => Some(value),
            _ => None,
        }
    }

    pub fn as_tuple(&self) -> Option<&[Value]> {
        match self {
            Self::Tuple(value) => Some(value),
            _ => None,
        }
    }

    pub fn as_map(&self) -> Option<&[(Value, Value)]> {
        match self {
            Self::Map(value) => Some(value),
            _ => None,
        }
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        self.as_map()?
            .iter()
            .find_map(|(candidate, value)| (candidate.as_str() == Some(key)).then_some(value))
    }

    pub fn pretty(&self) -> String {
        let mut out = String::new();
        self.write_pretty(&mut out, 0, 4);
        out
    }

    fn write_pretty(&self, out: &mut String, depth: usize, max_depth: usize) {
        if depth >= max_depth {
            out.push('…');
            return;
        }
        match self {
            Self::None => out.push_str("null"),
            Self::Bool(value) => out.push_str(if *value { "true" } else { "false" }),
            Self::Int(value) => out.push_str(&value.to_string()),
            Self::Float(value) => out.push_str(&value.to_string()),
            Self::String(value) => {
                out.push('"');
                for character in value.chars() {
                    match character {
                        '"' => out.push_str("\\\""),
                        '\\' => out.push_str("\\\\"),
                        '\n' => out.push_str("\\n"),
                        '\r' => out.push_str("\\r"),
                        '\t' => out.push_str("\\t"),
                        other => out.push(other),
                    }
                }
                out.push('"');
            }
            Self::Bytes(value) => out.push_str(&format!("<{} bytes>", value.len())),
            Self::List(values) | Self::Tuple(values) => {
                let (open, close) = if matches!(self, Self::List(_)) {
                    ('[', ']')
                } else {
                    ('(', ')')
                };
                out.push(open);
                for (index, value) in values.iter().enumerate() {
                    if index > 0 {
                        out.push_str(", ");
                    }
                    value.write_pretty(out, depth + 1, max_depth);
                }
                out.push(close);
            }
            Self::Map(entries) => {
                out.push('{');
                for (index, (key, value)) in entries.iter().enumerate() {
                    if index > 0 {
                        out.push_str(", ");
                    }
                    key.write_pretty(out, depth + 1, max_depth);
                    out.push_str(": ");
                    value.write_pretty(out, depth + 1, max_depth);
                }
                out.push('}');
            }
            Self::Global { module, name } => out.push_str(&format!("<global {module}.{name}>")),
            Self::Persistent(value) => {
                out.push_str("<persistent ");
                value.write_pretty(out, depth + 1, max_depth);
                out.push('>');
            }
            Self::Storage(storage) => out.push_str(&format!(
                "<storage {} {:?} {} elements>",
                storage.key, storage.dtype, storage.elements
            )),
            Self::Tensor(tensor) => {
                out.push_str(&format!("<tensor {:?} {:?}>", tensor.dtype, tensor.shape))
            }
            Self::Reduce { callable, args } => {
                out.push_str("<reduce ");
                callable.write_pretty(out, depth + 1, max_depth);
                out.push(' ');
                args.write_pretty(out, depth + 1, max_depth);
                out.push('>');
            }
        }
    }
}
