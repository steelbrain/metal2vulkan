//! Values and their encoding (`docs/REPLAY-STREAM.md`, sections 1 and 2): unsigned LEB128 varints,
//! zigzag for signed ones, and the tagged values a call's arguments and answers hold.

use std::fmt;
use std::sync::Arc;

/// A blob's name: the SHA-256 of its data.
pub type Hash = [u8; 32];

/// One value of a call's arguments or answers. A map's keys and a call's names are interned
/// strings ([`Name`]).
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Nil,
    Bool(bool),
    Uint(u64),
    Sint(i64),
    F32(f32),
    F64(f64),
    /// An object, by its stream id (never 0: nil is [`Value::Nil`]).
    Object(u64),
    /// A GPU address, which the player relocates.
    Address(u64),
    Str(String),
    Bytes(Vec<u8>),
    Blob(Hash),
    List(Vec<Value>),
    Map(Vec<(Name, Value)>),
}

/// An interned name: a selector, a map key or an answer key.
pub type Name = Arc<str>;

const TAG_NIL: u8 = 0;
const TAG_FALSE: u8 = 1;
const TAG_TRUE: u8 = 2;
const TAG_UINT: u8 = 3;
const TAG_SINT: u8 = 4;
const TAG_F32: u8 = 5;
const TAG_F64: u8 = 6;
const TAG_OBJECT: u8 = 7;
const TAG_ADDRESS: u8 = 8;
const TAG_STRING: u8 = 9;
const TAG_BYTES: u8 = 10;
const TAG_BLOB: u8 = 11;
const TAG_LIST: u8 = 12;
const TAG_MAP: u8 = 13;

impl Value {
    pub fn as_uint(&self) -> Option<u64> {
        match self {
            Value::Uint(v) => Some(*v),
            _ => None,
        }
    }

    /// The object's id; `None` for nil and for every other kind.
    pub fn as_object(&self) -> Option<u64> {
        match self {
            Value::Object(id) => Some(*id),
            _ => None,
        }
    }

    pub fn as_address(&self) -> Option<u64> {
        match self {
            Value::Address(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(v) => Some(v),
            _ => None,
        }
    }

    /// Inline data; a blob is resolved to bytes before a backend sees a call.
    pub fn as_bytes(&self) -> Option<&[u8]> {
        match self {
            Value::Bytes(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_list(&self) -> Option<&[Value]> {
        match self {
            Value::List(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_map(&self) -> Option<&[(Name, Value)]> {
        match self {
            Value::Map(v) => Some(v),
            _ => None,
        }
    }

    /// A list of unsigned integers (a struct such as `MTLSize`).
    pub fn as_uints(&self) -> Option<Vec<u64>> {
        self.as_list()?.iter().map(Value::as_uint).collect()
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Nil => write!(f, "nil"),
            Value::Bool(v) => write!(f, "{v}"),
            Value::Uint(v) => write!(f, "{v}"),
            Value::Sint(v) => write!(f, "{v:+}"),
            Value::F32(v) => write!(f, "{v}f"),
            Value::F64(v) => write!(f, "{v}d"),
            Value::Object(id) => write!(f, "#{id}"),
            Value::Address(v) => write!(f, "@{v:#x}"),
            Value::Str(v) => write!(f, "{v:?}"),
            Value::Bytes(v) => write!(f, "<{} bytes>", v.len()),
            Value::Blob(hash) => write!(f, "blob:{}", hex(hash)),
            Value::List(items) => {
                write!(f, "[")?;
                for (at, item) in items.iter().enumerate() {
                    if at > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{item}")?;
                }
                write!(f, "]")
            }
            Value::Map(entries) => {
                write!(f, "{{")?;
                for (at, (key, item)) in entries.iter().enumerate() {
                    if at > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{key}: {item}")?;
                }
                write!(f, "}}")
            }
        }
    }
}

/// `bytes` in lowercase hex.
pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

pub(super) fn put_varint(out: &mut Vec<u8>, mut v: u64) {
    loop {
        let byte = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            out.push(byte);
            return;
        }
        out.push(byte | 0x80);
    }
}

/// Encode `value`, its map keys through `intern`.
pub(super) fn put_value(out: &mut Vec<u8>, value: &Value, intern: &mut impl FnMut(&str) -> u64) {
    match value {
        Value::Nil => out.push(TAG_NIL),
        Value::Bool(false) => out.push(TAG_FALSE),
        Value::Bool(true) => out.push(TAG_TRUE),
        Value::Uint(v) => {
            out.push(TAG_UINT);
            put_varint(out, *v);
        }
        Value::Sint(v) => {
            out.push(TAG_SINT);
            put_varint(out, ((v << 1) ^ (v >> 63)) as u64);
        }
        Value::F32(v) => {
            out.push(TAG_F32);
            out.extend_from_slice(&v.to_le_bytes());
        }
        Value::F64(v) => {
            out.push(TAG_F64);
            out.extend_from_slice(&v.to_le_bytes());
        }
        Value::Object(id) => {
            out.push(TAG_OBJECT);
            put_varint(out, *id);
        }
        Value::Address(v) => {
            out.push(TAG_ADDRESS);
            put_varint(out, *v);
        }
        Value::Str(v) => {
            out.push(TAG_STRING);
            put_varint(out, v.len() as u64);
            out.extend_from_slice(v.as_bytes());
        }
        Value::Bytes(v) => {
            out.push(TAG_BYTES);
            put_varint(out, v.len() as u64);
            out.extend_from_slice(v);
        }
        Value::Blob(hash) => {
            out.push(TAG_BLOB);
            out.extend_from_slice(hash);
        }
        Value::List(items) => {
            out.push(TAG_LIST);
            put_varint(out, items.len() as u64);
            for item in items {
                put_value(out, item, intern);
            }
        }
        Value::Map(entries) => {
            out.push(TAG_MAP);
            put_varint(out, entries.len() as u64);
            for (key, item) in entries {
                put_varint(out, intern(key));
                put_value(out, item, intern);
            }
        }
    }
}

/// A cursor over one record's payload.
pub(super) struct Payload<'a> {
    bytes: &'a [u8],
    at: usize,
}

/// Why a payload does not decode.
pub(super) type Bad = String;

impl<'a> Payload<'a> {
    pub(super) fn new(bytes: &'a [u8]) -> Self {
        Payload { bytes, at: 0 }
    }

    pub(super) fn done(&self) -> bool {
        self.at == self.bytes.len()
    }

    pub(super) fn remaining(&self) -> usize {
        self.bytes.len() - self.at
    }

    pub(super) fn take(&mut self, n: usize) -> Result<&'a [u8], Bad> {
        let end = self
            .at
            .checked_add(n)
            .filter(|&end| end <= self.bytes.len());
        let end = end.ok_or_else(|| format!("{n} bytes past the payload's end"))?;
        let taken = &self.bytes[self.at..end];
        self.at = end;
        Ok(taken)
    }

    fn byte(&mut self) -> Result<u8, Bad> {
        Ok(self.take(1)?[0])
    }

    pub(super) fn varint(&mut self) -> Result<u64, Bad> {
        let mut v = 0u64;
        for shift in (0..70).step_by(7) {
            let byte = self.byte()?;
            let bits = u64::from(byte & 0x7f);
            if shift == 63 && bits > 1 {
                return Err("a varint past 64 bits".into());
            }
            v |= bits << shift;
            if byte & 0x80 == 0 {
                return Ok(v);
            }
        }
        Err("a varint longer than 10 bytes".into())
    }

    fn length(&mut self) -> Result<usize, Bad> {
        let n = self.varint()?;
        usize::try_from(n).map_err(|_| format!("a length of {n}"))
    }

    /// One value, its map keys resolved through `name`.
    pub(super) fn value(&mut self, name: &impl Fn(u64) -> Result<Name, Bad>) -> Result<Value, Bad> {
        Ok(match self.byte()? {
            TAG_NIL => Value::Nil,
            TAG_FALSE => Value::Bool(false),
            TAG_TRUE => Value::Bool(true),
            TAG_UINT => Value::Uint(self.varint()?),
            TAG_SINT => {
                let v = self.varint()?;
                Value::Sint(((v >> 1) as i64) ^ -((v & 1) as i64))
            }
            TAG_F32 => Value::F32(f32::from_le_bytes(self.take(4)?.try_into().unwrap())),
            TAG_F64 => Value::F64(f64::from_le_bytes(self.take(8)?.try_into().unwrap())),
            TAG_OBJECT => match self.varint()? {
                0 => return Err("object 0".into()),
                id => Value::Object(id),
            },
            TAG_ADDRESS => Value::Address(self.varint()?),
            TAG_STRING => {
                let n = self.length()?;
                let text = std::str::from_utf8(self.take(n)?).map_err(|e| e.to_string())?;
                Value::Str(text.to_string())
            }
            TAG_BYTES => {
                let n = self.length()?;
                Value::Bytes(self.take(n)?.to_vec())
            }
            TAG_BLOB => Value::Blob(self.take(32)?.try_into().unwrap()),
            TAG_LIST => {
                let n = self.length()?;
                let mut items = Vec::with_capacity(n.min(self.bytes.len()));
                for _ in 0..n {
                    items.push(self.value(name)?);
                }
                Value::List(items)
            }
            TAG_MAP => {
                let n = self.length()?;
                let mut entries: Vec<(Name, Value)> = Vec::with_capacity(n.min(self.bytes.len()));
                for _ in 0..n {
                    let key = name(self.varint()?)?;
                    if entries.iter().any(|(seen, _)| *seen == key) {
                        return Err(format!("the map key {key} twice"));
                    }
                    let item = self.value(name)?;
                    entries.push((key, item));
                }
                Value::Map(entries)
            }
            tag => return Err(format!("value tag {tag}")),
        })
    }
}
