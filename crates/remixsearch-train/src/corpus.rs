use crate::diagrams::draw_diagram;
use remixsearch_format::{Document, ImageRef, QueryRecord};
use std::fs;
use std::path::{Path, PathBuf};

pub struct CorpusPaths {
    pub docs: PathBuf,
    pub queries: PathBuf,
    pub images: PathBuf,
}

const SUBJECTS: &[(&str, &str)] = &[
    ("blocks", "independent brotli blocks"),
    ("hnsw", "hierarchical navigable graphs"),
    ("distill", "asymmetric query distillation"),
    ("figures", "image retrieval"),
    ("bm25", "lexical postings"),
    ("tokens", "byte-level vocabularies"),
    ("astro", "static site hooks"),
    ("int8", "quantized query encoders"),
];

pub fn generate_corpus(dir: &Path, concepts_per_subject: usize, draw: bool) -> std::io::Result<CorpusPaths> {
    let images = dir.join("images");
    fs::create_dir_all(&images)?;
    let mut docs = Vec::new();
    let mut queries = Vec::new();
    for (subject, title) in SUBJECTS {
        let how_url = format!("/{subject}/how-it-works#a");
        docs.push(Document {
            url: how_url.clone(),
            title: format!("how {subject} works"),
            headings: vec![(*title).to_string(), format!("how {subject} works")],
            text: format!(
                "How {subject} works. {subject} is {title}. It works by giving every signal its own scale and its own window. Working through {subject} means reading those two parameters from that signal's reference page. How {subject} is working is never answered by a different signal, and a comparison page only says the scales differ."
            ),
            images: vec![ImageRef {
                src: format!("images/{subject}-0.png"),
                alt: format!("diagram of how {subject} works"),
                caption: format!("{subject} works by a scale and a window on each signal"),
            }],
        });
        for q in [
            format!("how does {subject} work"),
            format!("how does the {subject} work"),
            format!("how does the {subject} working"),
            format!("how is {subject} working"),
            format!("how {subject} works"),
            format!("what is {subject}?"),
        ] {
            queries.push(QueryRecord {
                query: q,
                positive_url: how_url.clone(),
                kind: "passage".into(),
                held_out: false,
                unanswerable: false,
            });
        }
        queries.push(QueryRecord {
            query: format!("explain how the {subject} is working"),
            positive_url: how_url,
            kind: "passage".into(),
            held_out: true,
            unanswerable: false,
        });
        for i in 0..concepts_per_subject {
            let name = format!("{subject} signal {i}");
            let scale = 0.15 + i as f32 * 0.017;
            let window = 32 + (i % 7) * 8;
            let other = (i + 3) % concepts_per_subject.max(1);
            let fact = format!(
                "In the {title} notes, {name} uses scale {scale:.3} and window {window}. A question about {name} must resolve here, not to {subject} signal {other}."
            );
            let contrast = format!(
                "{subject} signal {other} uses a different scale and must not answer questions about {name}."
            );
            let held = i % 10 == 0;
            let image_name = format!("{subject}-{i}.png");
            let image_path = images.join(&image_name);
            if draw {
                draw_diagram(&image_path, &name, i as u32)?;
            }
            let image = ImageRef {
                src: format!("images/{image_name}"),
                alt: format!("diagram of {name}"),
                caption: format!("{name} at scale {scale:.3}"),
            };
            let pages = [
                ("overview", format!("{fact} The overview walks through the build step that writes this fact into the index.")),
                ("reference", format!("Reference for {name}. Parameter scale is {scale:.3}. Parameter window is {window}. {fact}")),
                ("faq", format!("FAQ: what is {name}? {fact}")),
                ("troubleshooting", format!("If search misses {name}, the window {window} was not stored. {contrast}")),
                ("comparison", format!("Compare {name} with {subject} signal {other}. {contrast} {fact}")),
                ("figure", format!("The figure for {name} shows scale {scale:.3}. {fact}")),
            ];
            for (kind, body) in pages {
                let url = format!("/{subject}/{i}/{kind}");
                let page_title = format!("{name} {kind}");
                let part_a = format!("{body} Part A records the primary fact for {page_title}.");
                let part_b = format!("Part B of {page_title} adds the operational detail: reload block {window} after editing scale {scale:.3}. {contrast}");
                for (part, text) in [("a", part_a), ("b", part_b)] {
                    let section_url = format!("{url}#{part}");
                    let images = if kind == "figure" && part == "a" { vec![image.clone()] } else { Vec::new() };
                    docs.push(Document {
                        url: section_url.clone(),
                        title: format!("{page_title} {part}"),
                        headings: vec![title.to_string(), page_title.clone()],
                        text,
                        images,
                    });
                    let section_title = format!("{page_title} {part}");
                    let mut variants = vec![
                        section_title.clone(),
                        format!("explain {section_title}"),
                        format!("{subject} {kind} for signal {i}"),
                    ];
                    if kind == "reference" && part == "a" {
                        variants.push(format!("what is {name}?"));
                        variants.push(format!("how does {name} work"));
                        variants.push(format!("scale {scale:.3} window {window}"));
                    }
                    for q in variants {
                        queries.push(QueryRecord {
                            query: q,
                            positive_url: section_url.clone(),
                            kind: "passage".into(),
                            held_out: held,
                            unanswerable: false,
                        });
                    }
                }
                if kind == "figure" {
                    queries.push(QueryRecord {
                        query: format!("diagram of {name}"),
                        positive_url: image.src.clone(),
                        kind: "image".into(),
                        held_out: held,
                        unanswerable: false,
                    });
                    queries.push(QueryRecord {
                        query: format!("figure showing scale {scale:.3} for {name}"),
                        positive_url: image.src.clone(),
                        kind: "image".into(),
                        held_out: held,
                        unanswerable: false,
                    });
                }
            }
        }
    }
    for n in 0..40 {
        queries.push(QueryRecord {
            query: format!("migration route of monarch butterfly cohort {n}"),
            positive_url: String::new(),
            kind: "passage".into(),
            held_out: true,
            unanswerable: true,
        });
    }
    let docs_path = dir.join("corpus.jsonl");
    let queries_path = dir.join("queries.jsonl");
    write_jsonl(&docs_path, &docs)?;
    write_jsonl(&queries_path, &queries)?;
    Ok(CorpusPaths { docs: docs_path, queries: queries_path, images })
}

fn write_jsonl<T: serde::Serialize>(path: &Path, rows: &[T]) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = fs::File::create(path)?;
    for row in rows {
        serde_json::to_writer(&mut f, row).map_err(std::io::Error::other)?;
        f.write_all(b"\n")?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_corpus_has_distinct_sections_and_a_diagram() {
        let dir = tempfile::tempdir().unwrap();
        let paths = generate_corpus(dir.path(), 2, true).unwrap();
        let docs = std::fs::read_to_string(&paths.docs).unwrap();
        let lines = docs.lines().count();
        // one how-it-works page plus 2 concepts * 6 pages * 2 parts, for 8 subjects
        assert_eq!(lines, 8 * (1 + 2 * 6 * 2));
        assert!(docs.contains("scale 0.150"));
        assert!(docs.contains("scale 0.167"));
        assert!(docs.contains("How bm25 works"));
        assert!(paths.images.join("blocks-0.png").exists());
        let queries = std::fs::read_to_string(&paths.queries).unwrap();
        assert!(queries.contains("how does the bm25 working"));
        assert!(queries.contains("diagram of blocks signal 0"));
        assert!(queries.contains("unanswerable"));
    }
}
