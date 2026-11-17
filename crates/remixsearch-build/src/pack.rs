use crate::ingest::Chunk;
use remixsearch_format::{
    encode_graph, encode_meta, encode_vectors, quantize, Manifest, NodeKind, NodeMeta, Package,
    FORMAT_VERSION, PACKAGE_FILE,
};
use remixsearch_infer::{l2_normalize, HnswGraph};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

pub struct PackedIndex {
    pub manifest: Manifest,
}

pub fn pack_index(
    chunks: &[Chunk],
    vectors: &[Vec<f32>],
    student_bytes: &[u8],
    tokenizer_bytes: &[u8],
    teacher: &str,
    output: &Path,
) -> std::io::Result<PackedIndex> {
    assert_eq!(chunks.len(), vectors.len(), "one vector per chunk");
    let mut vecs = vectors.to_vec();
    for v in &mut vecs {
        l2_normalize(v);
    }
    let graph = HnswGraph::build(&vecs, 16, 200, 42);
    let order = locality_order(&graph);
    let mut chunks_o = Vec::with_capacity(chunks.len());
    let mut vecs_o = Vec::with_capacity(vecs.len());
    for id in &order {
        chunks_o.push(chunks[*id as usize].clone());
        vecs_o.push(vecs[*id as usize].clone());
    }
    let remap = {
        let mut m = vec![0u32; order.len()];
        for (new_id, old) in order.iter().enumerate() {
            m[*old as usize] = new_id as u32;
        }
        m
    };
    let level0: Vec<Vec<u32>> = (0..order.len())
        .map(|new_id| {
            let old = order[new_id];
            graph.level0[old as usize].iter().map(|n| remap[*n as usize]).collect()
        })
        .collect();
    let upper: Vec<Vec<Vec<u32>>> = graph
        .upper
        .iter()
        .map(|layer| {
            (0..order.len())
                .map(|new_id| {
                    let old = order[new_id];
                    layer[old as usize].iter().map(|n| remap[*n as usize]).collect()
                })
                .collect()
        })
        .collect();
    let entry = remap[graph.entry as usize];

    fs::create_dir_all(output)?;
    for stale in ["vectors", "graph", "meta", "lexical", "model"] {
        let path = output.join(stale);
        if path.is_dir() {
            fs::remove_dir_all(path)?;
        }
    }
    for stale in ["manifest.json", "tokenizer.br"] {
        let path = output.join(stale);
        if path.is_file() {
            fs::remove_file(path)?;
        }
    }

    let mut family_image: HashMap<String, (String, String, String)> = HashMap::new();
    for chunk in &chunks_o {
        if let Some(image) = chunk.images.first() {
            family_image.entry(concept_key(&chunk.url)).or_insert_with(|| {
                (image.src.clone(), image.alt.clone(), image.caption.clone())
            });
        }
    }
    let quantized: Vec<_> = vecs_o.iter().map(|v| quantize(v)).collect();
    let metas: Vec<NodeMeta> = chunks_o
        .iter()
        .map(|c| {
            let own = c.images.first();
            let inherited = family_image.get(&concept_key(&c.url));
            let image_url = own.map(|i| i.src.clone()).or_else(|| inherited.map(|i| i.0.clone())).unwrap_or_default();
            let alt = own.map(|i| i.alt.clone()).or_else(|| inherited.map(|i| i.1.clone())).unwrap_or_default();
            let caption = own.map(|i| i.caption.clone()).or_else(|| inherited.map(|i| i.2.clone())).unwrap_or_default();
            let kind = if c.text.is_empty() && own.is_some() { NodeKind::Image } else { NodeKind::Passage };
            NodeMeta {
                kind,
                url: c.url.clone(),
                title: c.title.clone(),
                image_url,
                alt: alt.clone(),
                caption: caption.clone(),
                snippet: if c.text.trim().is_empty() {
                    format!("{alt} {caption}")
                } else {
                    c.text.chars().take(700).collect()
                },
            }
        })
        .collect();

    let manifest = Manifest {
        version: FORMAT_VERSION,
        dims: remixsearch_format::DIMS,
        doc_count: chunks_o.len() as u32,
        entry_id: entry,
        max_level: upper.len() as u32,
        m: 16,
        ef_search: 64,
        teacher: teacher.to_string(),
        upper_layers: upper,
    };
    let mut package = Package::new();
    package.insert("manifest", serde_json::to_vec(&manifest).expect("manifest"));
    package.insert("vectors", encode_vectors(&quantized));
    package.insert("graph", encode_graph(&level0));
    package.insert("meta", encode_meta(&metas));
    package.insert("model", student_bytes.to_vec());
    package.insert("tokenizer", tokenizer_bytes.to_vec());
    let bytes = package.compress().expect("brotli");
    fs::write(output.join(PACKAGE_FILE), bytes)?;
    Ok(PackedIndex { manifest })
}

fn concept_key(url: &str) -> String {
    let path = url.split('#').next().unwrap_or(url);
    let parts: Vec<&str> = path.split('/').filter(|p| !p.is_empty()).collect();
    if parts.len() >= 2 {
        format!("{}/{}", parts[0], parts[1])
    } else {
        parts.join("/")
    }
}

fn locality_order(graph: &HnswGraph) -> Vec<u32> {
    let n = graph.level0.len();
    let mut seen = vec![false; n];
    let mut order = Vec::with_capacity(n);
    let mut stack = vec![graph.entry];
    while let Some(id) = stack.pop() {
        if seen[id as usize] {
            continue;
        }
        seen[id as usize] = true;
        order.push(id);
        let mut neigh = graph.level0[id as usize].clone();
        neigh.sort_unstable();
        for nb in neigh.into_iter().rev() {
            if !seen[nb as usize] {
                stack.push(nb);
            }
        }
    }
    for i in 0..n as u32 {
        if !seen[i as usize] {
            order.push(i);
        }
    }
    order
}
