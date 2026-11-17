//! On-disk layout for a search index.
//!
//! Vectors, the HNSW graph, passage text, the query model, and the tokenizer
//! sit in one solid brotli package (`index.br`), the same way a zip keeps its
//! members in a single file. One compression context covers every member.

mod codec;

pub use codec::*;

use serde::{Deserialize, Serialize};

pub const DIMS: usize = 128;
pub const BLOCK_DOCS: usize = 32;
pub const FORMAT_VERSION: u32 = 2;
pub const PACKAGE_FILE: &str = "index.br";

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    Message(String),
    #[error("brotli: {0}")]
    Brotli(String),
}

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum NodeKind {
    Passage,
    Image,
}

impl NodeKind {
    pub fn from_u8(v: u8) -> Self {
        if v == 1 { Self::Image } else { Self::Passage }
    }
    pub fn as_u8(self) -> u8 {
        match self {
            Self::Passage => 0,
            Self::Image => 1,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NodeMeta {
    pub kind: NodeKind,
    pub url: String,
    pub title: String,
    pub image_url: String,
    pub alt: String,
    pub caption: String,
    pub snippet: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ImageRef {
    pub src: String,
    pub alt: String,
    #[serde(default)]
    pub caption: String,
}

/// One input record. A build system can emit this as JSONL.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Document {
    pub url: String,
    pub title: String,
    #[serde(default)]
    pub headings: Vec<String>,
    pub text: String,
    #[serde(default)]
    pub images: Vec<ImageRef>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QueryRecord {
    pub query: String,
    pub positive_url: String,
    #[serde(default)]
    pub kind: String,
    #[serde(default)]
    pub held_out: bool,
    #[serde(default)]
    pub unanswerable: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Manifest {
    pub version: u32,
    pub dims: usize,
    pub doc_count: u32,
    pub entry_id: u32,
    pub max_level: u32,
    pub m: u32,
    pub ef_search: usize,
    pub teacher: String,
    /// Neighbors for levels above 0, indexed by node id. Empty slots are unused.
    pub upper_layers: Vec<Vec<Vec<u32>>>,
}

#[derive(Clone, Debug)]
pub struct QuantVector {
    pub scale: f32,
    pub values: Vec<i8>,
}

pub fn term_hash(term: &str) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for b in term.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

pub fn lexical_tokens(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() > 1)
        .map(|t| t.to_lowercase())
        .collect()
}

pub fn quantize(v: &[f32]) -> QuantVector {
    let max = v.iter().map(|x| x.abs()).fold(0f32, f32::max);
    let scale = if max < 1e-12 { 1.0 } else { max / 127.0 };
    let values = v
        .iter()
        .map(|x| (x / scale).round().clamp(-127.0, 127.0) as i8)
        .collect();
    QuantVector { scale, values }
}

pub fn dequantize(v: &QuantVector) -> Vec<f32> {
    v.values.iter().map(|x| *x as f32 * v.scale).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn brotli_roundtrip_and_quantize() {
        let raw = b"neighbor lists repeat small integers".repeat(20);
        let packed = brotli_compress(&raw).unwrap();
        assert!(packed.len() < raw.len());
        assert_eq!(brotli_decompress(&packed).unwrap(), raw);

        let chunk = b"neighbor lists repeat small integers".repeat(40);
        let mut package = Package::new();
        package.insert("vectors", chunk.clone());
        package.insert("graph", chunk.clone());
        let solid = package.compress().unwrap();
        let separate = brotli_compress(&chunk).unwrap().len() * 2;
        assert!(solid.len() < separate, "solid {} separate {separate}", solid.len());
        let opened = Package::open(&solid).unwrap();
        assert_eq!(opened.get("vectors").unwrap(), chunk.as_slice());

        let q = quantize(&[0.0, 0.5, -1.0, 0.25]);
        let back = dequantize(&q);
        assert!((back[2] + 1.0).abs() < 0.02);
    }
}
