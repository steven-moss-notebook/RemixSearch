//! Turn HTML, Markdown, or JSONL into one brotli package of an HNSW index.

mod ingest;
mod pack;

pub use ingest::{expand_images, ingest_path, synthesize_questions, Chunk};
pub use pack::{pack_index, PackedIndex};

use remixsearch_format::{Document, QueryRecord};
use std::path::Path;

pub fn read_queries(path: &Path) -> std::io::Result<Vec<QueryRecord>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let text = std::fs::read_to_string(path)?;
    let trimmed = text.trim();
    if trimmed.starts_with('[') {
        return Ok(serde_json::from_str(trimmed).unwrap_or_default());
    }
    let mut out = Vec::new();
    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        if let Ok(q) = serde_json::from_str::<QueryRecord>(line) {
            out.push(q);
        }
    }
    Ok(out)
}

pub fn read_jsonl(path: &Path) -> std::io::Result<Vec<Document>> {
    let text = std::fs::read_to_string(path)?;
    let mut out = Vec::new();
    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        if let Ok(doc) = serde_json::from_str::<Document>(line) {
            out.push(doc);
        }
    }
    Ok(out)
}
