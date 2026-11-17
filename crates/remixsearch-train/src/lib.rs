//! Corpus generation, tokenizer training, the EmbeddingGemma 2 teacher, and
//! distillation into a Pico-sized query encoder.

mod corpus;
mod diagrams;
mod distill;
mod gemma;
mod math;
mod student;
mod tokenizer;

pub use corpus::{generate_corpus, CorpusPaths};
pub use distill::{distill, DistillConfig, HashTeacher, MIN_CONTRAST};
pub use gemma::{cached_teacher_dir, GemmaTeacher, EMBEDDING_GEMMA_REPO};
pub use student::QueryEncoder;
pub use tokenizer::train_tokenizer;

use remixsearch_build::Chunk;
use remixsearch_format::DIMS;
use remixsearch_infer::l2_normalize;

/// Bag-of-tokens embedding used when EmbeddingGemma weights are not on disk.
/// Questions that share terms land near the same passage.
pub fn hash_embed(text: &str) -> Vec<f32> {
    let mut v = vec![0f32; DIMS];
    for token in remixsearch_format::lexical_tokens(text) {
        let h = remixsearch_format::term_hash(&token);
        let bin = (h as usize) % DIMS;
        let sign = if (h >> 17) & 1 == 0 { 1.0 } else { -1.0 };
        v[bin] += sign;
        v[(bin + 3) % DIMS] += sign * 0.5;
    }
    l2_normalize(&mut v);
    v
}

pub fn embed_chunk_hash(chunk: &Chunk) -> Vec<f32> {
    let mut text = format!("{} {}", chunk.title, chunk.text);
    for image in &chunk.images {
        text.push(' ');
        text.push_str(&image.alt);
        text.push(' ');
        text.push_str(&image.caption);
    }
    hash_embed(&text)
}
