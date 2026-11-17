//! Query-time search. This crate compiles to WASM: no Burn, no filesystem.

mod bpe;
mod hnsw;
mod student;

pub use bpe::CompactBpe;
pub use hnsw::{cosine, l2_normalize, search_layer, HnswGraph};
pub use student::{LayerWeights, StudentF32, StudentWeights};

use remixsearch_format::{
    decode_graph, decode_meta, decode_vectors, dequantize, lexical_tokens, Error, Manifest, NodeKind, NodeMeta,
    Package, Result, DIMS,
};
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct Hit {
    pub id: u32,
    pub kind: NodeKind,
    pub title: String,
    pub url: String,
    pub image_url: String,
    pub alt: String,
    /// Cosine that produced this order, from -1 to 1.
    pub score: f32,
    /// Cosine between the query vector and this node, from -1 to 1.
    pub cosine: f32,
    /// Query words that occur in the title, snippet, alt, or caption.
    pub matched: Vec<String>,
    pub snippet: String,
}

#[derive(Clone, Debug)]
pub struct TermFact {
    pub term: String,
    pub pieces: Vec<String>,
    pub unknown: bool,
}

#[derive(Clone, Debug)]
pub struct SearchReport {
    pub terms: Vec<TermFact>,
    pub candidates: usize,
    pub best_cosine: f32,
    pub second_cosine: f32,
    pub confident: bool,
    pub summary: String,
}

#[derive(Clone, Debug)]
pub struct SearchOutcome {
    pub hits: Vec<Hit>,
    pub report: SearchReport,
}

pub struct Searcher {
    pub manifest: Manifest,
    vectors: HashMap<u32, Vec<f32>>,
    neighbors: HashMap<u32, Vec<u32>>,
    meta: HashMap<u32, NodeMeta>,
    student: StudentF32,
    tokenizer: CompactBpe,
}

impl Searcher {
    pub fn open(package: &Package) -> Result<Self> {
        let manifest: Manifest = serde_json::from_slice(package.get("manifest")?)
            .map_err(|e| Error::Message(e.to_string()))?;
        let vectors = decode_vectors(package.get("vectors")?)?
            .into_iter()
            .enumerate()
            .map(|(i, q)| (i as u32, dequantize(&q)))
            .collect();
        let neighbors = decode_graph(package.get("graph")?)?
            .into_iter()
            .enumerate()
            .map(|(i, n)| (i as u32, n))
            .collect();
        let meta = decode_meta(package.get("meta")?)?
            .into_iter()
            .enumerate()
            .map(|(i, m)| (i as u32, m))
            .collect();
        let student = StudentF32::from_int8(package.get("model")?)?;
        let tokenizer = CompactBpe::from_bytes(package.get("tokenizer")?)?;
        Ok(Self { manifest, vectors, neighbors, meta, student, tokenizer })
    }

    pub fn vector(&self, id: u32) -> Option<&[f32]> {
        self.vectors.get(&id).map(|v| v.as_slice())
    }

    pub fn node(&self, id: u32) -> Option<&NodeMeta> {
        self.meta.get(&id)
    }

    pub fn encode_query(&self, query: &str) -> Vec<f32> {
        let ids = self.tokenizer.encode(query);
        self.student.embed(&ids)
    }

    pub fn search(&self, query: &str, k: usize) -> SearchOutcome {
        let q = self.encode_query(query);
        let vector = |id: u32| self.vectors.get(&id).cloned().unwrap_or_else(|| vec![0.0; DIMS]);
        let mut ep = self.manifest.entry_id;
        let max_level = self.manifest.max_level as usize;
        for level in (1..=max_level).rev() {
            let layer = &self.manifest.upper_layers[level - 1];
            let neighbors = |id: u32| layer.get(id as usize).cloned().unwrap_or_default();
            if let Some((id, _)) = search_layer(&q, ep, 1, &neighbors, &vector).first() {
                ep = *id;
            }
        }
        let neighbors = |id: u32| self.neighbors.get(&id).cloned().unwrap_or_default();
        let mut found = search_layer(&q, ep, self.manifest.ef_search.max(k), &neighbors, &vector);
        let facts = self.term_facts(query);
        let best_cosine = found.first().map(|(_, cos)| *cos).unwrap_or(0.0);
        let second_cosine = found.get(1).map(|(_, cos)| *cos).unwrap_or(best_cosine);
        let candidates = found.len();
        let (summary, confident) = diagnosis(&facts, candidates, best_cosine, second_cosine);
        found.truncate(k);
        let hits = found
            .into_iter()
            .filter_map(|(id, cos)| {
                let meta = self.meta.get(&id)?;
                let (snippet, matched) = explain(query, meta);
                Some(Hit {
                    id,
                    kind: meta.kind,
                    title: meta.title.clone(),
                    url: meta.url.clone(),
                    image_url: meta.image_url.clone(),
                    alt: meta.alt.clone(),
                    score: cos,
                    cosine: cos,
                    matched,
                    snippet,
                })
            })
            .collect();
        SearchOutcome {
            hits,
            report: SearchReport {
                terms: facts,
                candidates,
                best_cosine,
                second_cosine,
                confident,
                summary,
            },
        }
    }

    fn term_facts(&self, query: &str) -> Vec<TermFact> {
        content_terms(query)
            .into_iter()
            .map(|term| {
                let pieces = self.tokenizer.pieces(&term);
                let unknown = pieces.is_empty() || pieces.iter().all(|(_, id)| *id == self.tokenizer.unk_id);
                TermFact {
                    term,
                    pieces: pieces.into_iter().map(|(text, _)| text.replace('Ġ', "·")).collect(),
                    unknown,
                }
            })
            .collect()
    }
}

fn content_terms(query: &str) -> Vec<String> {
    let mut terms = lexical_tokens(query);
    for raw in query.split(|c: char| !c.is_alphanumeric()) {
        let lower = raw.to_lowercase();
        if !lower.is_empty() && lower.chars().all(|c| c.is_ascii_digit()) && !terms.iter().any(|t| t == &lower) {
            terms.push(lower);
        }
    }
    terms.retain(|term| {
        !matches!(
            term.as_str(),
            "the" | "and" | "for" | "what" | "how" | "with" | "this" | "that" | "from" | "are"
                | "was" | "were" | "you" | "your" | "does" | "about" | "is" | "a" | "an" | "do"
                | "i" | "in" | "of" | "on" | "to" | "it" | "or" | "at" | "by" | "be" | "as" | "if"
                | "so" | "not" | "into" | "my" | "me" | "we" | "can" | "up"
        )
    });
    terms
}

fn diagnosis(terms: &[TermFact], candidates: usize, best: f32, second: f32) -> (String, bool) {
    let gap = best - second;
    let unknown: Vec<&str> = terms.iter().filter(|t| t.unknown).map(|t| t.term.as_str()).collect();
    let mut lines = Vec::new();
    if terms.is_empty() {
        lines.push("The query has no usable words after dropping fillers.".into());
    } else {
        let words: Vec<&str> = terms.iter().map(|t| t.term.as_str()).collect();
        lines.push(format!("Query words: {}.", words.join(", ")));
    }
    if !unknown.is_empty() {
        lines.push(format!(
            "{} {} no tokenizer merge, so the encoder sees unknown byte pieces.",
            quote_list(&unknown),
            if unknown.len() == 1 { "has" } else { "have" }
        ));
    }
    let split: Vec<String> = terms
        .iter()
        .filter(|t| !t.unknown && t.pieces.len() > 1)
        .map(|t| format!("'{}' into {}", t.term, t.pieces.join(" ")))
        .collect();
    if !split.is_empty() {
        lines.push(format!(
            "The tokenizer split {}, so the encoder has no single vector for the whole word.",
            split.join("; ")
        ));
    }
    lines.push(format!(
        "Nearest cosine is {best:.3}; the next is {second:.3} (gap {gap:.3}) across {candidates} candidates. Cosine 0 is unrelated and 1 is the same direction."
    ));
    let confident = best >= 0.25 || gap >= 0.05;
    if gap < 0.05 && best < 0.35 {
        lines.push("The candidates sit on top of each other, so this order is noise. A larger gap after more training of the query encoder is what changes the list.".into());
    } else if confident {
        lines.push("The top cosine is separated from the next candidate.".into());
    }
    (lines.join(" "), confident)
}

fn quote_list(words: &[&str]) -> String {
    words.iter().map(|w| format!("'{w}'")).collect::<Vec<_>>().join(", ")
}

fn explain(query: &str, meta: &NodeMeta) -> (String, Vec<String>) {
    let hay = format!("{} {} {}", meta.snippet, meta.alt, meta.caption);
    let terms = content_terms(query);
    let lower: Vec<char> = hay.to_lowercase().chars().collect();
    let title_lower: Vec<char> = meta.title.to_lowercase().chars().collect();
    let chars: Vec<char> = hay.chars().collect();
    let mut matched = Vec::new();
    let mut first: Option<usize> = None;
    for term in terms {
        let needle: Vec<char> = term.chars().collect();
        if needle.is_empty() {
            continue;
        }
        let in_body = find_chars(&lower, &needle);
        let in_title = find_chars(&title_lower, &needle);
        if in_body.is_some() || in_title.is_some() {
            if !matched.iter().any(|m: &String| m == &term) {
                matched.push(term);
            }
            if let Some(pos) = in_body {
                first = Some(first.map(|p| p.min(pos)).unwrap_or(pos));
            }
        }
    }
    let snippet = match first {
        Some(pos) => window(&chars, pos, 70, 140),
        None => chars.iter().take(220).collect(),
    };
    (snippet, matched)
}

fn find_chars(hay: &[char], needle: &[char]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

fn window(chars: &[char], pos: usize, before: usize, after: usize) -> String {
    let start = pos.saturating_sub(before);
    let end = (pos + after).min(chars.len());
    let mut out = String::new();
    if start > 0 {
        out.push('…');
    }
    out.extend(chars[start..end].iter().copied());
    if end < chars.len() {
        out.push('…');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use remixsearch_format::{quantize, NodeKind};

    fn unit(seed: u32) -> Vec<f32> {
        let mut v = vec![0f32; DIMS];
        let mut x = seed as f32 + 1.0;
        for s in &mut v {
            x = (x * 1.37 + 0.13).sin();
            *s = x;
        }
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
        for s in &mut v {
            *s /= n;
        }
        v
    }

    #[test]
    fn function_words_are_not_query_terms() {
        let terms = content_terms("how do i setup a search index in astro");
        assert!(terms.iter().any(|t| t == "setup"));
        assert!(terms.iter().any(|t| t == "astro"));
        assert!(terms.iter().any(|t| t == "search"));
        assert!(!terms.iter().any(|t| matches!(t.as_str(), "do" | "i" | "in" | "a" | "how")));
    }

    #[test]
    fn snippet_centers_on_the_matched_word() {
        use remixsearch_format::NodeMeta;
        let meta = NodeMeta {
            kind: NodeKind::Passage,
            url: "/d".into(),
            title: "scale notes".into(),
            image_url: String::new(),
            alt: String::new(),
            caption: String::new(),
            snippet: format!("{}the window parameter is 48", "padding ".repeat(30)),
        };
        let (snippet, matched) = explain("what window is stored", &meta);
        assert_eq!(matched, vec!["window".to_string()]);
        assert!(snippet.to_lowercase().contains("window"));
        assert!(snippet.starts_with('…'));
    }

    #[test]
    fn an_absent_word_is_described_as_noise() {
        let facts = vec![TermFact { term: "test".into(), pieces: vec!["test".into()], unknown: false }];
        let (summary, confident) = diagnosis(&facts, 8, 0.12, 0.10);
        assert!(!confident);
        assert!(summary.contains("0.12"));
        assert!(summary.contains("noise"));
    }

    #[test]
    fn packaged_hnsw_matches_memory() {
        let vecs: Vec<Vec<f32>> = (0..40).map(unit).collect();
        let graph = HnswGraph::build(&vecs, 8, 64, 7);
        let package = pack_memory(&vecs, &graph);
        let searcher = Searcher::open(&package).unwrap();
        let hits = search_layer(
            &vecs[3],
            searcher.manifest.entry_id,
            32,
            &|id| searcher.neighbors.get(&id).cloned().unwrap_or_default(),
            &|id| searcher.vectors.get(&id).cloned().unwrap_or_else(|| vec![0.0; DIMS]),
        );
        let memory = search_layer(
            &vecs[3],
            graph.entry,
            32,
            &|id| graph.level0[id as usize].clone(),
            &|id| vecs[id as usize].clone(),
        );
        assert_eq!(hits[0].0, memory[0].0, "packaged {hits:?} memory {memory:?}");
    }

    fn pack_memory(vecs: &[Vec<f32>], graph: &HnswGraph) -> remixsearch_format::Package {
        use remixsearch_format::{encode_graph, encode_meta, encode_vectors, Manifest, NodeMeta, Package};
        let manifest = Manifest {
            version: 2,
            dims: DIMS,
            doc_count: vecs.len() as u32,
            entry_id: graph.entry,
            max_level: graph.upper.len() as u32,
            m: 8,
            ef_search: 32,
            teacher: "test".into(),
            upper_layers: graph.upper.clone(),
        };
        let metas: Vec<_> = (0..vecs.len())
            .map(|i| NodeMeta {
                kind: NodeKind::Passage,
                url: format!("/d/{i}"),
                title: format!("Doc {i}"),
                image_url: String::new(),
                alt: String::new(),
                caption: String::new(),
                snippet: format!("snippet {i}"),
            })
            .collect();
        let mut package = Package::new();
        package.insert("manifest", serde_json::to_vec(&manifest).unwrap());
        package.insert("vectors", encode_vectors(&vecs.iter().map(|v| quantize(v)).collect::<Vec<_>>()));
        package.insert("graph", encode_graph(&graph.level0));
        package.insert("meta", encode_meta(&metas));
        package.insert("model", StudentF32::identity_for_tests().to_int8());
        package.insert("tokenizer", CompactBpe::byte_fallback_for_tests().to_bytes());
        package
    }
}
