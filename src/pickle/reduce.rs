use crate::{DType, Error, Result, StorageRef, TensorMeta, Value};

use super::literal::type_error;

pub(super) fn reduce(callable: Value, args: Value, offset: usize) -> Result<Value> {
    if global_is(&callable, "collections", "OrderedDict") {
        return Ok(Value::Map(Vec::new()));
    }
    if global_is(&callable, "torch._utils", "_rebuild_tensor_v2")
        || global_is(&callable, "torch._utils", "_rebuild_tensor")
    {
        return rebuild_tensor(args, offset);
    }
    if global_is(&callable, "torch._utils", "_rebuild_parameter") {
        let values = tuple(args, offset)?;
        return values
            .into_iter()
            .next()
            .ok_or_else(|| Error::pickle(offset, "empty _rebuild_parameter arguments"));
    }
    Ok(Value::Reduce {
        callable: Box::new(callable),
        args: Box::new(args),
    })
}

pub(super) fn persistent(id: Value, offset: usize) -> Result<Value> {
    let values = match id {
        Value::Tuple(values) => values,
        other => return Ok(Value::Persistent(Box::new(other))),
    };
    if values.first().and_then(Value::as_str) != Some("storage") || values.len() < 5 {
        return Ok(Value::Persistent(Box::new(Value::Tuple(values))));
    }
    let (module, name) = match &values[1] {
        Value::Global { module, name } => (module.as_str(), name.as_str()),
        other => return Err(type_error(offset, "storage type global", other.kind())),
    };
    let dtype = DType::from_storage_global(module, name).ok_or_else(|| {
        Error::pickle(offset, format!("unsupported storage type {module}.{name}"))
    })?;
    let key = scalar_string(&values[2], offset)?;
    let location = scalar_string(&values[3], offset)?;
    let elements = nonnegative(&values[4], offset, "storage size")?;
    Ok(Value::Storage(StorageRef {
        key,
        dtype,
        location,
        elements,
    }))
}

fn rebuild_tensor(args: Value, offset: usize) -> Result<Value> {
    let values = tuple(args, offset)?;
    if values.len() < 4 {
        return Err(Error::pickle(
            offset,
            "tensor rebuild has fewer than four arguments",
        ));
    }
    let storage = match &values[0] {
        Value::Storage(storage) => storage.clone(),
        other => return Err(type_error(offset, "storage", other.kind())),
    };
    let storage_offset = nonnegative(&values[1], offset, "storage offset")?;
    let shape = dimensions(&values[2], offset)?;
    let stride = dimensions(&values[3], offset)?;
    let requires_grad = values.get(4).and_then(Value::as_bool).unwrap_or(false);
    let tensor = TensorMeta {
        dtype: storage.dtype,
        storage,
        storage_offset,
        shape,
        stride,
        requires_grad,
    };
    let span = tensor.storage_span_elements()?;
    if tensor
        .storage_offset
        .checked_add(span)
        .filter(|end| *end <= tensor.storage.elements)
        .is_none()
    {
        return Err(Error::pickle(offset, "tensor view exceeds its storage"));
    }
    Ok(Value::Tensor(tensor))
}

fn dimensions(value: &Value, offset: usize) -> Result<Vec<u64>> {
    let values = match value {
        Value::Tuple(values) | Value::List(values) => values,
        other => return Err(type_error(offset, "dimension tuple", other.kind())),
    };
    values
        .iter()
        .map(|value| nonnegative(value, offset, "dimension"))
        .collect()
}
fn tuple(value: Value, offset: usize) -> Result<Vec<Value>> {
    match value {
        Value::Tuple(values) => Ok(values),
        other => Err(type_error(offset, "tuple", other.kind())),
    }
}
fn nonnegative(value: &Value, offset: usize, what: &str) -> Result<u64> {
    let value = value
        .as_i64()
        .ok_or_else(|| type_error(offset, "integer", value.kind()))?;
    u64::try_from(value).map_err(|_| Error::pickle(offset, format!("{what} is negative")))
}
fn scalar_string(value: &Value, offset: usize) -> Result<String> {
    match value {
        Value::String(value) => Ok(value.clone()),
        Value::Bytes(value) => String::from_utf8(value.clone())
            .map_err(|_| Error::pickle(offset, "storage field is not UTF-8")),
        other => Err(type_error(offset, "string", other.kind())),
    }
}
fn global_is(value: &Value, expected_module: &str, expected_name: &str) -> bool {
    matches!(value, Value::Global { module, name } if module == expected_module && name == expected_name)
}
