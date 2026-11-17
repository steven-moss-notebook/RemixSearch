use remixsearch_infer::CompactBpe;
use std::path::Path;
use tokenizers::models::bpe::{BpeTrainer, BPE};
use tokenizers::models::TrainerWrapper;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::{AddedToken, Tokenizer};

pub fn train_tokenizer(texts: &[String], vocab_size: usize) -> Result<CompactBpe, String> {
    let dir = std::env::temp_dir().join(format!("remixsearch-tok-{}", std::process::id()));
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let corpus = dir.join("corpus.txt");
    std::fs::write(&corpus, texts.join("\n")).map_err(|e| e.to_string())?;
    let specials = ["<pad>", "<unk>", "<cls>", "<sep>"]
        .into_iter()
        .map(|s| AddedToken::from(s, true))
        .collect::<Vec<_>>();
    let mut trainer = TrainerWrapper::BpeTrainer(
        BpeTrainer::builder()
            .vocab_size(vocab_size)
            .min_frequency(2)
            .special_tokens(specials)
            .build(),
    );
    let model = BPE::builder().unk_token("<unk>".into()).build().map_err(|e| e.to_string())?;
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(ByteLevel::default()));
    tokenizer
        .train_from_files(&mut trainer, vec![corpus.to_string_lossy().into_owned()])
        .map_err(|e| e.to_string())?;
    let json_path = dir.join("tokenizer.json");
    tokenizer.save(&json_path, false).map_err(|e| e.to_string())?;
    let json = std::fs::read_to_string(&json_path).map_err(|e| e.to_string())?;
    let value: serde_json::Value = serde_json::from_str(&json).map_err(|e| e.to_string())?;
    compact_from_json(&value).map_err(|e| e.to_string())
}

fn compact_from_json(value: &serde_json::Value) -> Result<CompactBpe, String> {
    let vocab_obj = value["model"]["vocab"]
        .as_object()
        .ok_or("tokenizer json has no vocab")?;
    let mut id_to_token = vec![String::new(); vocab_obj.len()];
    for (token, id) in vocab_obj {
        let id = id.as_u64().ok_or("vocab id")? as usize;
        if id >= id_to_token.len() {
            id_to_token.resize(id + 1, String::new());
        }
        id_to_token[id] = token.clone();
    }
    let mut vocab = std::collections::HashMap::new();
    for (id, token) in id_to_token.iter().enumerate() {
        vocab.insert(token.clone(), id as u32);
    }
    let mut merges = std::collections::HashMap::new();
    if let Some(rows) = value["model"]["merges"].as_array() {
        for (rank, row) in rows.iter().enumerate() {
            let pair = if let Some(s) = row.as_str() {
                s.split_once(' ').map(|(a, b)| (a.to_string(), b.to_string()))
            } else if let Some(arr) = row.as_array() {
                Some((
                    arr.first().and_then(|v| v.as_str()).unwrap_or("").to_string(),
                    arr.get(1).and_then(|v| v.as_str()).unwrap_or("").to_string(),
                ))
            } else {
                None
            };
            if let Some(pair) = pair {
                merges.insert(pair, rank);
            }
        }
    }
    let unk_id = vocab.get("<unk>").copied().unwrap_or(1);
    let pad_id = vocab.get("<pad>").copied().unwrap_or(0);
    Ok(CompactBpe { vocab, id_to_token, merges, unk_id, pad_id })
}

pub fn save_tokenizer(tok: &CompactBpe, path: &Path) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, tok.to_bytes())
}

pub fn load_tokenizer(path: &Path) -> Result<CompactBpe, String> {
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    CompactBpe::from_bytes(&bytes).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokenizers::Tokenizer;

    #[test]
    fn trained_bpe_matches_tokenizers_crate() {
        let mut texts = Vec::new();
        for i in 0..80 {
            texts.push(format!(
                "brotli block {i} stores neighbor lists for the hnsw layer and the image caption of signal {i}"
            ));
        }
        let compact = train_tokenizer(&texts, 400).unwrap();
        let dir = std::env::temp_dir().join(format!("remixsearch-tok-{}", std::process::id()));
        let reference = Tokenizer::from_file(dir.join("tokenizer.json")).unwrap();
        for text in ["what is brotli block 3?", "hnsw layer image caption", "unknown ζητα word"] {
            let ours = compact.encode(text);
            let theirs = reference.encode(text, false).unwrap().get_ids().to_vec();
            assert_eq!(ours, theirs, "encode mismatch for {text:?}");
        }
        let bytes = compact.to_bytes();
        let loaded = CompactBpe::from_bytes(&bytes).unwrap();
        assert_eq!(loaded.encode("hnsw layer"), compact.encode("hnsw layer"));
    }
}
