use image::ImageReader;
use pulldown_cmark::{Event, Options, Parser, Tag, TagEnd};
use scraper::{Html, Selector};
use remixsearch_format::{Document, ImageRef, QueryRecord};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
pub struct Chunk {
    pub url: String,
    pub title: String,
    pub headings: Vec<String>,
    pub text: String,
    pub images: Vec<ImageRef>,
}

pub fn ingest_path(path: &Path) -> std::io::Result<Vec<Chunk>> {
    if path.is_file() {
        if path.extension().and_then(|e| e.to_str()) == Some("jsonl") {
            let docs = crate::read_jsonl(path)?;
            return Ok(docs.into_iter().flat_map(|d| chunk_document(&d)).collect());
        }
        let text = std::fs::read_to_string(path)?;
        return Ok(chunk_file(path, &text, path.parent().unwrap_or(path)));
    }
    let mut chunks = Vec::new();
    let mut files = Vec::new();
    collect(path, &mut files);
    for file in files {
        if file.extension().and_then(|e| e.to_str()) == Some("jsonl") {
            if let Ok(docs) = crate::read_jsonl(&file) {
                chunks.extend(docs.into_iter().flat_map(|d| chunk_document(&d)));
            }
            continue;
        }
        let Ok(text) = std::fs::read_to_string(&file) else { continue };
        chunks.extend(chunk_file(&file, &text, path));
    }
    Ok(chunks)
}

fn collect(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for entry in rd.flatten() {
        let p = entry.path();
        if p.is_dir() {
            if p.file_name().and_then(|s| s.to_str()) == Some("node_modules") {
                continue;
            }
            collect(&p, out);
        } else if matches!(p.extension().and_then(|e| e.to_str()), Some("html" | "htm" | "md" | "jsonl")) {
            out.push(p);
        }
    }
}

fn chunk_file(path: &Path, text: &str, root: &Path) -> Vec<Chunk> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    let rel = path.strip_prefix(root).unwrap_or(path);
    let url = format!("/{}", rel.to_string_lossy().replace('\\', "/"));
    if ext == "md" {
        chunk_markdown(&url, text, path.parent())
    } else if ext == "html" || ext == "htm" {
        chunk_html(&url, text, path.parent())
    } else {
        Vec::new()
    }
}

fn chunk_document(doc: &Document) -> Vec<Chunk> {
    let mut chunks = chunk_text(&doc.url, &doc.title, &doc.headings, &doc.text);
    if let Some(first) = chunks.first_mut() {
        first.images = doc.images.clone();
    } else if !doc.images.is_empty() {
        chunks.push(Chunk {
            url: doc.url.clone(),
            title: doc.title.clone(),
            headings: doc.headings.clone(),
            text: doc.text.clone(),
            images: doc.images.clone(),
        });
    }
    chunks
}

pub fn chunk_markdown(url: &str, text: &str, base: Option<&Path>) -> Vec<Chunk> {
    let mut options = Options::empty();
    options.insert(Options::ENABLE_TABLES);
    let parser = Parser::new_ext(text, options);
    let mut headings = Vec::new();
    let mut current_title = String::new();
    let mut body = String::new();
    let mut images = Vec::new();
    let mut in_heading = false;
    let mut sections: Vec<(String, Vec<String>, String, Vec<ImageRef>)> = Vec::new();
    let flush = |title: &mut String, headings: &mut Vec<String>, body: &mut String, images: &mut Vec<ImageRef>, sections: &mut Vec<_>| {
        if body.trim().is_empty() && images.is_empty() {
            return;
        }
        sections.push((title.clone(), headings.clone(), body.trim().to_string(), images.clone()));
        body.clear();
        images.clear();
    };
    for event in parser {
        match event {
            Event::Start(Tag::Heading { .. }) => {
                flush(&mut current_title, &mut headings, &mut body, &mut images, &mut sections);
                in_heading = true;
                current_title.clear();
            }
            Event::End(TagEnd::Heading(_)) => {
                in_heading = false;
                if !current_title.is_empty() {
                    headings.push(current_title.clone());
                    if headings.len() > 6 {
                        headings.remove(0);
                    }
                }
            }
            Event::Start(Tag::Image { dest_url, .. }) => {
                let src = resolve_src(base, &dest_url);
                images.push(ImageRef { src, alt: String::new(), caption: String::new() });
            }
            Event::Text(t) => {
                if in_heading {
                    current_title.push_str(&t);
                } else {
                    if let Some(img) = images.last_mut() {
                        if img.alt.is_empty() {
                            img.alt = t.to_string();
                        }
                    }
                    body.push_str(&t);
                    body.push(' ');
                }
            }
            Event::SoftBreak | Event::HardBreak => body.push('\n'),
            _ => {}
        }
    }
    flush(&mut current_title, &mut headings, &mut body, &mut images, &mut sections);
    let title = sections.first().map(|s| s.0.clone()).unwrap_or_else(|| "page".into());
    sections
        .into_iter()
        .enumerate()
        .flat_map(|(i, (heading, hs, text, images))| {
            let anchor = if heading.is_empty() { url.to_string() } else { format!("{url}#{i}") };
            let page_title = if heading.is_empty() { title.clone() } else { heading };
            chunk_text_with_images(&anchor, &page_title, &hs, &text, images)
        })
        .collect()
}

pub fn chunk_html(url: &str, text: &str, base: Option<&Path>) -> Vec<Chunk> {
    let html = Html::parse_document(text);
    let title = html
        .select(&Selector::parse("title").unwrap())
        .next()
        .map(|n| n.text().collect::<String>())
        .unwrap_or_else(|| "page".into());
    let Some(node) = content_node(&html) else {
        return Vec::new();
    };
    let mut plain = String::new();
    for t in node.text() {
        let t = t.trim();
        if !t.is_empty() {
            plain.push_str(t);
            plain.push(' ');
        }
    }
    let mut images = Vec::new();
    let img_sel = Selector::parse("img").unwrap();
    for img in node.select(&img_sel) {
        let src = img.value().attr("src").unwrap_or("");
        if src.is_empty() {
            continue;
        }
        images.push(ImageRef {
            src: resolve_src(base, src),
            alt: img.value().attr("alt").unwrap_or("").to_string(),
            caption: String::new(),
        });
    }
    chunk_text_with_images(url, title.trim(), &[], plain.trim(), images)
}

fn content_node(html: &Html) -> Option<scraper::ElementRef<'_>> {
    for selector in ["#main-content", "main", "article", "body"] {
        let parsed = Selector::parse(selector).ok()?;
        if let Some(node) = html.select(&parsed).next() {
            return Some(node);
        }
    }
    None
}

fn resolve_src(base: Option<&Path>, src: &str) -> String {
    if src.starts_with("http://") || src.starts_with("https://") || src.starts_with('/') {
        return src.to_string();
    }
    if let Some(base) = base {
        return base.join(src).to_string_lossy().to_string();
    }
    src.to_string()
}

fn chunk_text(url: &str, title: &str, headings: &[String], text: &str) -> Vec<Chunk> {
    chunk_text_with_images(url, title, headings, text, Vec::new())
}

fn chunk_text_with_images(url: &str, title: &str, headings: &[String], text: &str, images: Vec<ImageRef>) -> Vec<Chunk> {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() && images.is_empty() {
        return Vec::new();
    }
    if words.len() <= 180 {
        return vec![Chunk {
            url: url.to_string(),
            title: title.to_string(),
            headings: headings.to_vec(),
            text: words.join(" "),
            images,
        }];
    }
    let mut chunks = Vec::new();
    let mut start = 0;
    let mut part = 0;
    while start < words.len() {
        let end = (start + 160).min(words.len());
        let mut c = Chunk {
            url: format!("{url}#p{part}"),
            title: format!("{title}"),
            headings: headings.to_vec(),
            text: words[start..end].join(" "),
            images: Vec::new(),
        };
        if part == 0 {
            c.images = images.clone();
        }
        chunks.push(c);
        if end == words.len() {
            break;
        }
        start = end.saturating_sub(40);
        part += 1;
    }
    chunks
}

/// Split figures out of a passage so each image is its own index node.
pub fn expand_images(chunks: Vec<Chunk>) -> Vec<Chunk> {
    let mut out = Vec::new();
    for chunk in chunks {
        let images = chunk.images.clone();
        if !chunk.text.trim().is_empty() {
            out.push(chunk.clone());
        }
        for image in images {
            out.push(Chunk {
                url: chunk.url.clone(),
                title: if image.alt.is_empty() { chunk.title.clone() } else { image.alt.clone() },
                headings: chunk.headings.clone(),
                text: String::new(),
                images: vec![image],
            });
        }
    }
    out
}

pub fn synthesize_questions(chunks: &[Chunk]) -> Vec<QueryRecord> {
    let mut queries = Vec::new();
    for chunk in chunks {
        let name = item_name(&chunk.title);
        if name.len() < 3 {
            continue;
        }
        let variants = [
            (format!("how does {name} work"), false),
            (format!("how do I use {name}"), false),
            (format!("what is {name}?"), false),
            (format!("explain how {name} works"), true),
        ];
        for (q, held_out) in variants {
            queries.push(QueryRecord {
                query: q,
                positive_url: chunk.url.clone(),
                kind: "passage".into(),
                held_out,
                unanswerable: false,
            });
        }
        for image in &chunk.images {
            let label = if image.alt.is_empty() { name.as_str() } else { image.alt.as_str() };
            queries.push(QueryRecord {
                query: format!("diagram of {label}"),
                positive_url: image.src.clone(),
                kind: "image".into(),
                held_out: false,
                unanswerable: false,
            });
        }
    }
    queries
}

fn item_name(title: &str) -> String {
    let title = title.trim().strip_suffix(" - Rust").unwrap_or(title.trim());
    if let Some((name, _)) = title.rsplit_once(" in ") {
        if !name.contains(' ') && !name.is_empty() {
            return name.replace('_', " ");
        }
    }
    title.to_string()
}

pub fn image_size(path: &Path) -> Option<(u32, u32)> {
    let reader = ImageReader::open(path).ok()?;
    let (w, h) = reader.into_dimensions().ok()?;
    Some((w, h))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn markdown_chunks_keep_images() {
        let md = "# Brotli blocks\n\nEach block is independent.\n\n![layout](layout.png)\n\n## Graph\n\nNeighbors are varints.\n";
        let chunks = chunk_markdown("/docs/blocks", md, None);
        assert!(chunks.len() >= 2);
        assert!(chunks.iter().any(|c| c.images.iter().any(|i| i.src.ends_with("layout.png"))));
        let qs = synthesize_questions(&chunks);
        assert!(qs.iter().any(|q| q.kind == "image"));
    }

    #[test]
    fn html_index_keeps_the_doc_and_drops_the_chrome() {
        let html = "<html><body><a>Skip to main content</a><nav>rustwide sidebar</nav><main><section id=\"main-content\"><h1>Searcher</h1><p>Runs a query over the HNSW index.</p></section></main></body></html>";
        let chunks = chunk_html("/remixsearch_infer/struct.Searcher.html", html, None);
        let text = chunks.iter().map(|c| c.text.clone()).collect::<Vec<_>>().join(" ");
        assert!(text.contains("HNSW"));
        assert!(!text.contains("Skip"));
        assert!(!text.contains("rustwide"));
    }

    #[test]
    fn questions_use_the_item_name_not_the_rustdoc_chrome() {
        let chunks = vec![Chunk {
            url: "/remixsearch_infer/struct.Searcher.html".into(),
            title: "Searcher in remixsearch_infer - Rust".into(),
            headings: Vec::new(),
            text: "Runs a query.".into(),
            images: Vec::new(),
        }];
        let qs = synthesize_questions(&chunks);
        assert!(qs.iter().any(|q| q.query == "how does Searcher work"));
        assert!(qs.iter().any(|q| q.held_out && q.query == "explain how Searcher works"));
        assert!(qs.iter().all(|q| !q.query.contains("remixsearch_infer")));
    }
}
