//! `remixsearch` builds a semantic index package and can search it from Rust.

use clap::{Parser, Subcommand};
use remixsearch_build::{expand_images, ingest_path, pack_index, read_queries, synthesize_questions, Chunk};
use remixsearch_format::{NodeKind, NodeMeta, Package, QueryRecord, PACKAGE_FILE};
use remixsearch_infer::{Searcher, StudentF32};
use remixsearch_train::{
    cached_teacher_dir, distill, embed_chunk_hash, generate_corpus, hash_embed, train_tokenizer, DistillConfig,
    GemmaTeacher,
};
use serde_json::json;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(name = "remixsearch", about = "Pack a site into a searchable remix")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Write a documentation-style corpus and labeled diagrams.
    GenerateCorpus {
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value_t = 50)]
        concepts: usize,
    },
    /// Train the byte-level BPE used by the query encoder.
    TrainTokenizer {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value_t = 8192)]
        vocab: usize,
    },
    /// Ingest pages, embed them, distill the query encoder, and pack one brotli package.
    Build {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: PathBuf,
        /// Local google/embeddinggemma-2 checkout. When omitted, the Hugging Face cache is used.
        #[arg(long)]
        teacher: Option<PathBuf>,
        /// Hash-embedding teacher instead of EmbeddingGemma 2.
        #[arg(long)]
        demo: bool,
        /// Also generate the bundled fact-bank corpus into the input directory.
        #[arg(long)]
        generate: bool,
        #[arg(long, default_value_t = 50)]
        concepts: usize,
        #[arg(long, default_value_t = 80)]
        epochs: usize,
        #[arg(long, default_value_t = 8192)]
        vocab: usize,
        #[arg(long, default_value_t = 4096)]
        max_pairs: usize,
    },
    /// Run one query against a packed index.
    Search {
        #[arg(long)]
        index: PathBuf,
        #[arg(long)]
        query: String,
        #[arg(long, default_value_t = 5)]
        k: usize,
    },
    /// Score the teacher and the distilled searcher on a query JSONL file.
    Eval {
        #[arg(long)]
        index: PathBuf,
        #[arg(long)]
        queries: PathBuf,
        #[arg(long)]
        teacher: Option<PathBuf>,
        #[arg(long)]
        demo: bool,
        #[arg(long)]
        report: Option<PathBuf>,
    },
    /// Distill one corpus with the hash embedder and with EmbeddingGemma 2.
    Compare {
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value_t = 1)]
        concepts: usize,
        #[arg(long, default_value_t = 80)]
        epochs: usize,
    },
}

fn main() {
    if let Err(err) = run() {
        eprintln!("remixsearch: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    match Cli::parse().cmd {
        Cmd::GenerateCorpus { output, concepts } => {
            let paths = generate_corpus(&output, concepts, true).map_err(|e| e.to_string())?;
            println!("wrote {} and {}", paths.docs.display(), paths.queries.display());
        }
        Cmd::TrainTokenizer { input, output, vocab } => {
            let chunks = ingest_path(&input).map_err(|e| e.to_string())?;
            let texts = texts_of(&chunks);
            let tok = train_tokenizer(&texts, vocab)?;
            if let Some(parent) = output.parent() {
                fs::create_dir_all(parent).map_err(|e| e.to_string())?;
            }
            fs::write(&output, tok.to_bytes()).map_err(|e| e.to_string())?;
            println!("wrote {}", output.display());
        }
        Cmd::Build { input, output, teacher, demo, generate, concepts, epochs, vocab, max_pairs } => {
            build(&input, &output, teacher.as_deref(), demo, generate, concepts, epochs, vocab, max_pairs)?;
        }
        Cmd::Search { index, query, k } => {
            let hits = search_index(&index, &query, k)?;
            println!("{}", serde_json::to_string_pretty(&hits).map_err(|e| e.to_string())?);
        }
        Cmd::Compare { output, concepts, epochs } => compare(&output, concepts, epochs)?,
        Cmd::Eval { index, queries, teacher, demo, report } => {
            let text = evaluate(&index, &queries, teacher.as_deref(), demo)?;
            if let Some(path) = report {
                if let Some(parent) = path.parent() {
                    fs::create_dir_all(parent).map_err(|e| e.to_string())?;
                }
                fs::write(&path, &text).map_err(|e| e.to_string())?;
                println!("wrote {}", path.display());
            } else {
                println!("{text}");
            }
        }
    }
    Ok(())
}

struct Embedded {
    chunks: Vec<Chunk>,
    vectors: Vec<Vec<f32>>,
    teacher_name: String,
    embed_query: Box<dyn Fn(&str) -> Vec<f32>>,
}

fn build(
    input: &Path,
    output: &Path,
    teacher: Option<&Path>,
    demo: bool,
    generate: bool,
    concepts: usize,
    epochs: usize,
    vocab: usize,
    max_pairs: usize,
) -> Result<(), String> {
    let teacher_dir = resolve_teacher(teacher, demo)?;
    let teacher = teacher_dir.as_deref();
    if generate {
        generate_corpus(&input.join("generated"), concepts, true).map_err(|e| e.to_string())?;
    }
    let mut chunks = if input.exists() {
        expand_images(ingest_path(input).map_err(|e| e.to_string())?)
    } else {
        Vec::new()
    };
    if chunks.is_empty() {
        return Err(format!("no documents under {}", input.display()));
    }
    let embedded = embed_chunks(input, &chunks, teacher, demo)?;
    chunks = embedded.chunks;
    let queries = load_queries(input, &chunks);
    let mut texts = texts_of(&chunks);
    texts.extend(queries.iter().map(|q| q.query.clone()));
    let tok = train_tokenizer(&texts, vocab)?;
    let config = DistillConfig {
        epochs,
        batch_size: remixsearch_train::MIN_CONTRAST,
        max_pairs,
        lr: 1e-3,
        vocab,
        artifact_dir: output.join("artifacts"),
    };
    let weights = distill(&chunks, &embedded.vectors, &queries, &tok, embedded.embed_query.as_ref(), &config)?;
    let student = StudentF32 { weights }.to_int8();
    fs::create_dir_all(output).map_err(|e| e.to_string())?;
    stage_images(&mut chunks, input, output)?;
    pack_index(&chunks, &embedded.vectors, &student, &tok.to_bytes(), &embedded.teacher_name, output)
        .map_err(|e| e.to_string())?;
    println!(
        "packed {} nodes into {} (teacher {})",
        chunks.len(),
        output.display(),
        embedded.teacher_name
    );
    Ok(())
}

fn resolve_teacher(teacher: Option<&Path>, demo: bool) -> Result<Option<PathBuf>, String> {
    if demo {
        return Ok(None);
    }
    if let Some(path) = teacher {
        return Ok(Some(path.to_path_buf()));
    }
    eprintln!("fetching google/embeddinggemma-2 into the Hugging Face cache if it is not already there");
    cached_teacher_dir().map(Some)
}

fn compare(output: &Path, concepts: usize, epochs: usize) -> Result<(), String> {
    fs::create_dir_all(output).map_err(|e| e.to_string())?;
    let corpus = output.join("corpus");
    if corpus.exists() {
        fs::remove_dir_all(&corpus).map_err(|e| e.to_string())?;
    }
    generate_corpus(&corpus, concepts, true).map_err(|e| e.to_string())?;
    let queries = corpus.join("queries.jsonl");
    let mut reports = serde_json::Map::new();
    for (name, demo) in [("hash", true), ("embeddinggemma2", false)] {
        let index = output.join(name);
        if index.exists() {
            fs::remove_dir_all(&index).map_err(|e| e.to_string())?;
        }
        eprintln!("distilling with {name}");
        build(&corpus, &index, None, demo, false, concepts, epochs, 8192, 4096)?;
        let text = evaluate(&index, &queries, None, demo)?;
        reports.insert(name.to_string(), serde_json::from_str(&text).map_err(|e| e.to_string())?);
    }
    println!("{}", serde_json::to_string_pretty(&serde_json::Value::Object(reports)).map_err(|e| e.to_string())?);
    Ok(())
}

fn embed_chunks(base: &Path, chunks: &[Chunk], teacher: Option<&Path>, demo: bool) -> Result<Embedded, String> {
    if demo {
        let vectors = chunks.iter().map(embed_chunk_hash).collect();
        return Ok(Embedded {
            chunks: chunks.to_vec(),
            vectors,
            teacher_name: "demo".into(),
            embed_query: Box::new(|q| hash_embed(q)),
        });
    }
    let dir = teacher.ok_or("pass --teacher or --demo")?;
    let model = GemmaTeacher::load(dir)?;
    let mut vectors = Vec::with_capacity(chunks.len());
    for chunk in chunks {
        let vector = if chunk.text.is_empty() {
            if let Some(image) = chunk.images.first() {
                let path = resolve_image(base, &image.src);
                match model.embed_image(&path, &chunk.title, &image.caption) {
                    Ok(v) => v,
                    Err(err) => {
                        eprintln!("image {}: {err}", path.display());
                        embed_chunk_hash(chunk)
                    }
                }
            } else {
                embed_chunk_hash(chunk)
            }
        } else {
            model.embed_document(&chunk.title, &chunk.text)?
        };
        vectors.push(vector);
    }
    Ok(Embedded {
        chunks: chunks.to_vec(),
        vectors,
        teacher_name: "gemma".into(),
        embed_query: Box::new(move |q| model.embed_query(q).unwrap_or_else(|_| hash_embed(q))),
    })
}

fn stage_images(chunks: &mut [Chunk], input: &Path, output: &Path) -> Result<(), String> {
    let media = output.join("media");
    fs::create_dir_all(&media).map_err(|e| e.to_string())?;
    let mut used: std::collections::HashMap<String, PathBuf> = std::collections::HashMap::new();
    for chunk in chunks {
        for image in &mut chunk.images {
            let path = resolve_image(input, &image.src);
            if !path.is_file() {
                continue;
            }
            let base = path.file_name().and_then(|s| s.to_str()).unwrap_or("image").to_string();
            let mut name = base.clone();
            let mut n = 1;
            while used.get(&name).is_some_and(|existing| existing != &path) {
                n += 1;
                name = format!("{n}-{base}");
            }
            if !used.contains_key(&name) {
                fs::copy(&path, media.join(&name)).map_err(|e| e.to_string())?;
                used.insert(name.clone(), path);
            }
            image.src = format!("media/{name}");
        }
    }
    Ok(())
}

fn resolve_image(base: &Path, src: &str) -> PathBuf {
    let path = PathBuf::from(src);
    if path.is_absolute() {
        return path;
    }
    let direct = base.join(&path);
    if direct.exists() {
        direct
    } else {
        base.join("generated").join(path)
    }
}

fn texts_of(chunks: &[Chunk]) -> Vec<String> {
    chunks
        .iter()
        .map(|c| format!("{} {} {}", c.title, c.text, c.images.iter().map(|i| i.alt.as_str()).collect::<Vec<_>>().join(" ")))
        .filter(|t| t.trim().len() > 2)
        .collect()
}

fn load_queries(input: &Path, chunks: &[Chunk]) -> Vec<QueryRecord> {
    let mut queries = Vec::new();
    for path in [
        input.join("queries.json"),
        input.join("queries.jsonl"),
        input.join("generated/queries.jsonl"),
    ] {
        let found = read_queries(&path).unwrap_or_default();
        if !found.is_empty() {
            queries = found;
            break;
        }
    }
    let mut extra = synthesize_questions(chunks);
    extra.retain(|q| !queries.iter().any(|kept| kept.query.eq_ignore_ascii_case(&q.query)));
    queries.append(&mut extra);
    queries
}

fn open_searcher(index: &Path) -> Result<Searcher, String> {
    let bytes = fs::read(index.join(PACKAGE_FILE)).map_err(|e| e.to_string())?;
    let package = Package::open(&bytes).map_err(|e| e.to_string())?;
    Searcher::open(&package).map_err(|e| e.to_string())
}

fn search_index(index: &Path, query: &str, k: usize) -> Result<serde_json::Value, String> {
    let searcher = open_searcher(index)?;
    let outcome = searcher.search(query, k);
    Ok(json!({
        "summary": outcome.report.summary,
        "confident": outcome.report.confident,
        "best_cosine": outcome.report.best_cosine,
        "second_cosine": outcome.report.second_cosine,
        "candidates": outcome.report.candidates,
        "terms": outcome.report.terms.iter().map(|t| json!({
            "term": t.term, "unknown": t.unknown, "pieces": t.pieces,
        })).collect::<Vec<_>>(),
        "hits": outcome.hits.iter().map(|h| json!({
        "kind": match h.kind { NodeKind::Image => "image", NodeKind::Passage => "passage" },
        "title": h.title,
        "url": h.url,
        "image_url": h.image_url,
        "alt": h.alt,
        "score": h.score,
        "cosine": h.cosine,
        "matched": h.matched,
        "snippet": h.snippet,
    })).collect::<Vec<_>>(),
    }))
}

fn evaluate(index: &Path, queries_path: &Path, teacher: Option<&Path>, demo: bool) -> Result<String, String> {
    let searcher = open_searcher(index)?;
    let (vectors, meta) = load_nodes(&searcher)?;
    let queries = read_queries(queries_path).map_err(|e| e.to_string())?;
    let gemma = if demo || searcher.manifest.teacher == "demo" {
        None
    } else {
        let dir = resolve_teacher(teacher, false)?.ok_or("EmbeddingGemma 2 was not resolved")?;
        Some(GemmaTeacher::load(&dir)?)
    };
    let mut semantic = Scores::default();
    let mut teacher_scores = Scores::default();
    let mut refused = 0.0;
    let mut unanswerable = 0.0;
    for query in queries.iter().filter(|q| q.held_out) {
        if query.unanswerable {
            unanswerable += 1.0;
            let hits = searcher.search(&query.query, 1).hits;
            if hits.first().map(|h| h.score).unwrap_or(0.0) < 0.45 {
                refused += 1.0;
            }
            continue;
        }
        let hits = searcher.search(&query.query, 5).hits;
        semantic.observe(query, hits.iter().position(|h| hit_matches(h.url.as_str(), h.image_url.as_str(), &meta, query)));
        let q_teacher = match &gemma {
            Some(model) => model.embed_query(&query.query)?,
            None => hash_embed(&query.query),
        };
        teacher_scores.observe(query, rank_of(&q_teacher, &vectors, &meta, query));
    }
    let report = json!({
        "teacher": teacher_scores.report(),
        "search": semantic.report(),
        "refusal_rate": rate(refused, unanswerable),
    });
    serde_json::to_string_pretty(&report).map_err(|e| e.to_string())
}

#[derive(Default)]
struct Scores {
    n: f32,
    hit1: f32,
    hit5: f32,
    mrr: f32,
    passage_n: f32,
    passage_hit: f32,
    image_n: f32,
    image_hit: f32,
}

impl Scores {
    fn observe(&mut self, query: &QueryRecord, rank: Option<usize>) {
        self.n += 1.0;
        if query.kind == "image" {
            self.image_n += 1.0;
            if rank.is_some() {
                self.image_hit += 1.0;
            }
        } else {
            self.passage_n += 1.0;
            if rank.is_some() {
                self.passage_hit += 1.0;
            }
        }
        if rank == Some(0) {
            self.hit1 += 1.0;
        }
        if let Some(rank) = rank {
            self.hit5 += 1.0;
            self.mrr += 1.0 / (rank as f32 + 1.0);
        }
    }
    fn report(&self) -> serde_json::Value {
        json!({
            "recall_at_1": rate(self.hit1, self.n),
            "recall_at_5": rate(self.hit5, self.n),
            "mrr": rate(self.mrr, self.n),
            "passage_recall_at_5": rate(self.passage_hit, self.passage_n),
            "image_recall_at_5": rate(self.image_hit, self.image_n),
        })
    }
}

fn hit_matches(url: &str, image: &str, _meta: &[NodeMeta], query: &QueryRecord) -> bool {
    url == query.positive_url || image == query.positive_url
}

fn rank_of(query: &[f32], vectors: &[Vec<f32>], meta: &[NodeMeta], record: &QueryRecord) -> Option<usize> {
    let mut scored: Vec<(usize, f32)> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let mut d = 0.0;
            for (a, b) in query.iter().zip(v) {
                d += a * b;
            }
            (i, d)
        })
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.iter().take(5).position(|(i, _)| {
        let node = &meta[*i];
        node.url == record.positive_url || node.image_url == record.positive_url
    })
}

fn load_nodes(searcher: &Searcher) -> Result<(Vec<Vec<f32>>, Vec<NodeMeta>), String> {
    let n = searcher.manifest.doc_count as usize;
    let mut vectors = Vec::with_capacity(n);
    let mut meta = Vec::with_capacity(n);
    for id in 0..n as u32 {
        vectors.push(searcher.vector(id).unwrap_or(&[]).to_vec());
        meta.push(searcher.node(id).cloned().unwrap_or(NodeMeta {
            kind: NodeKind::Passage,
            url: String::new(),
            title: String::new(),
            image_url: String::new(),
            alt: String::new(),
            caption: String::new(),
            snippet: String::new(),
        }));
    }
    Ok((vectors, meta))
}

fn rate(hit: f32, n: f32) -> f32 {
    if n == 0.0 { 0.0 } else { hit / n }
}
