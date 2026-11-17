//! EmbeddingGemma 2 text tower and Gemma 4 vision tower.
//!
//! Weights come from a local directory, or from the Hugging Face cache for
//! `google/embeddinggemma-2`. Only `language_model.*`, `vision_tower.*`, and
//! `embed_vision.*` are loaded. Compute stays in f32.

use crate::math::{aspect_size, patchify, rope_cos_sin};
use burn::module::{Param, ParamId};
use burn::nn::{
    Embedding, EmbeddingConfig, Gelu, Initializer, Linear, LinearConfig, RmsNorm, RmsNormConfig,
};
use burn::prelude::*;
use burn::tensor::activation::softmax;
use image::ImageReader;
use memmap2::Mmap;
use safetensors::tensor::SafeTensors;
use safetensors::Dtype;
use std::fs::File;
use std::path::Path;
use tokenizers::Tokenizer;

const HIDDEN: usize = 512;
const VOCAB: usize = 262_144;
const LAYERS: usize = 24;
const HEADS: usize = 4;
const IMAGE_TOKEN: u32 = 258_880;
const BOS: u32 = 2;
const EOS: u32 = 1;

pub const EMBEDDING_GEMMA_REPO: &str = "google/embeddinggemma-2";

/// Download `google/embeddinggemma-2` into the Hugging Face cache when the
/// snapshot is not already there, and return that snapshot directory.
pub fn cached_teacher_dir() -> Result<std::path::PathBuf, String> {
    let client = hf_hub::HFClientSync::new().map_err(|e| e.to_string())?;
    let dir = client
        .model("google", "embeddinggemma-2")
        .snapshot_download()
        .allow_patterns(vec!["model.safetensors".to_string(), "tokenizer.json".to_string()])
        .send()
        .map_err(|e| e.to_string())?;
    if !dir.join("model.safetensors").is_file() || !dir.join("tokenizer.json").is_file() {
        return Err(format!(
            "Hugging Face cache at {} is missing model.safetensors or tokenizer.json",
            dir.display()
        ));
    }
    Ok(dir)
}

pub struct GemmaTeacher {
    text: TextTower,
    vision: VisionTower,
    embed_vision: Linear,
    tokenizer: Tokenizer,
    device: Device,
}

struct TextTower {
    embed: Embedding,
    layers: Vec<TextLayer>,
    ple: Linear,
    ple_norm: RmsNorm,
    norm: RmsNorm,
    proj: Linear,
    gelu: Gelu,
}

#[derive(Module, Debug)]
struct TextLayer {
    input_ln: RmsNorm,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    post_attn: RmsNorm,
    pre_ff: RmsNorm,
    gate: Linear,
    up: Linear,
    down: Linear,
    post_ff: RmsNorm,
    ple_gate: Linear,
    ple_up: Linear,
    ple_norm: RmsNorm,
    scalar: Param<Tensor<1>>,
    head_dim: usize,
    kv_heads: usize,
    sliding: usize,
}

struct VisionTower {
    patch: Linear,
    pos: Vec<f32>,
    layers: Vec<VisionLayer>,
    gelu: Gelu,
}

#[derive(Module, Debug)]
struct VisionLayer {
    input_ln: RmsNorm,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    post_attn: RmsNorm,
    pre_ff: RmsNorm,
    gate: Linear,
    up: Linear,
    down: Linear,
    post_ff: RmsNorm,
}

struct Store {
    bytes: Mmap,
}

impl Store {
    fn open(path: &Path) -> Result<Self, String> {
        let file = File::open(path).map_err(|e| e.to_string())?;
        let bytes = unsafe { Mmap::map(&file).map_err(|e| e.to_string())? };
        Ok(Self { bytes })
    }
    fn get(&self, name: &str) -> Result<Vec<f32>, String> {
        let st = SafeTensors::deserialize(&self.bytes).map_err(|e| e.to_string())?;
        let t = st.tensor(name).map_err(|e| format!("{name}: {e}"))?;
        let shape = t.shape();
        let numel: usize = shape.iter().product();
        let raw = t.data();
        let mut out = Vec::with_capacity(numel);
        match t.dtype() {
            Dtype::BF16 => {
                for chunk in raw.chunks_exact(2) {
                    let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                    out.push(f32::from_bits((bits as u32) << 16));
                }
            }
            Dtype::F32 => {
                for chunk in raw.chunks_exact(4) {
                    out.push(f32::from_le_bytes(chunk.try_into().unwrap()));
                }
            }
            other => return Err(format!("{name} has dtype {other:?}")),
        }
        if out.len() != numel {
            return Err(format!("{name} length {} != {numel}", out.len()));
        }
        Ok(out)
    }
}

fn zeros_linear(input: usize, output: usize, device: &Device) -> Linear {
    LinearConfig::new(input, output)
        .with_bias(false)
        .with_initializer(Initializer::Zeros)
        .init(device)
}

fn zeros_rms(dim: usize, device: &Device) -> RmsNorm {
    RmsNormConfig::new(dim).with_epsilon(1e-6).init(device)
}

fn zeros_embed(rows: usize, cols: usize, device: &Device) -> Embedding {
    EmbeddingConfig::new(rows, cols).with_initializer(Initializer::Zeros).init(device)
}

fn set_linear(linear: &mut Linear, pytorch: &[f32], out_dim: usize, in_dim: usize, device: &Device) {
    let mut data = vec![0f32; in_dim * out_dim];
    for o in 0..out_dim {
        for i in 0..in_dim {
            data[i * out_dim + o] = pytorch[o * in_dim + i];
        }
    }
    let tensor = Tensor::<2>::from_data(TensorData::new(data, [in_dim, out_dim]), device);
    linear.weight = Param::initialized(ParamId::new(), tensor);
}

fn set_vec(target: &mut Param<Tensor<1>>, data: &[f32], device: &Device) {
    let tensor = Tensor::<1>::from_data(TensorData::new(data.to_vec(), [data.len()]), device);
    *target = Param::initialized(ParamId::new(), tensor);
}

fn set_embed(embed: &mut Embedding, data: &[f32], rows: usize, cols: usize, device: &Device) {
    let tensor = Tensor::<2>::from_data(TensorData::new(data.to_vec(), [rows, cols]), device);
    embed.weight = Param::initialized(ParamId::new(), tensor);
}

fn layer_kind(index: usize) -> (usize, usize, usize) {
    if matches!(index, 5 | 11 | 17 | 23) {
        (512, 1, 0)
    } else {
        (256, 2, 1)
    }
}

impl GemmaTeacher {
    pub fn load(dir: &Path) -> Result<Self, String> {
        let weights = dir.join("model.safetensors");
        let store = Store::open(&weights)?;
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| e.to_string())?;
        let device = Device::default();
        let mut teacher = Self {
            text: TextTower::new(&device),
            vision: VisionTower::new(&device),
            embed_vision: zeros_linear(768, HIDDEN, &device),
            tokenizer,
            device,
        };
        let _ = store;
        load_teacher_weights(&mut teacher, dir)?;
        Ok(teacher)
    }

    pub fn embed_query(&self, query: &str) -> Result<Vec<f32>, String> {
        self.embed_text(&format!("task: search result | query: {query}"))
    }

    pub fn embed_document(&self, title: &str, text: &str) -> Result<Vec<f32>, String> {
        let title = if title.is_empty() { "none" } else { title };
        self.embed_text(&format!("title: {title} | text: {text}"))
    }

    pub fn embed_image(&self, path: &Path, title: &str, context: &str) -> Result<Vec<f32>, String> {
        let (patches, grid_w, grid_h) = prepare_image(path)?;
        let soft = self.vision.forward(&patches, grid_w, grid_h, &self.device)?;
        let soft = rms_noscale(self.embed_vision.forward(soft));
        let prefix = format!("title: {title} | text: {context}");
        let mut ids = vec![BOS];
        ids.extend(self.encode_ids(&prefix)?);
        let n_soft = soft.dims()[1];
        let text_len = ids.len();
        ids.extend(std::iter::repeat(IMAGE_TOKEN).take(n_soft));
        ids.push(EOS);
        let mut embeds = self.text.token_embeds(&ids, &self.device);
        let start = text_len;
        embeds = scatter_soft(embeds, soft, start);
        self.text.forward_embeds(embeds)
    }

    fn embed_text(&self, text: &str) -> Result<Vec<f32>, String> {
        let mut ids = vec![BOS];
        ids.extend(self.encode_ids(text)?);
        ids.push(EOS);
        ids.truncate(512);
        let embeds = self.text.token_embeds(&ids, &self.device);
        self.text.forward_embeds(embeds)
    }

    fn encode_ids(&self, text: &str) -> Result<Vec<u32>, String> {
        let enc = self.tokenizer.encode(text, false).map_err(|e| e.to_string())?;
        Ok(enc.get_ids().to_vec())
    }
}

fn prepare_image(path: &Path) -> Result<(Vec<f32>, usize, usize), String> {
    let img = ImageReader::open(path).map_err(|e| e.to_string())?.decode().map_err(|e| e.to_string())?.to_rgb8();
    let (w, h) = img.dimensions();
    let (th, tw) = aspect_size(h, w, 16, 2520, 3);
    let resized = image::imageops::resize(&img, tw, th, image::imageops::FilterType::CatmullRom);
    let mut rgb = Vec::with_capacity((tw * th * 3) as usize);
    for p in resized.pixels() {
        rgb.extend([p[0] as f32 / 255.0, p[1] as f32 / 255.0, p[2] as f32 / 255.0]);
    }
    let (patches, _) = patchify(&rgb, tw, th, 16);
    Ok((patches, (tw / 16) as usize, (th / 16) as usize))
}

impl TextTower {
    fn new(device: &Device) -> Self {
        let mut layers = Vec::with_capacity(LAYERS);
        for i in 0..LAYERS {
            let (head_dim, kv_heads, sliding) = layer_kind(i);
            let q_out = HEADS * head_dim;
            let kv_out = kv_heads * head_dim;
            layers.push(TextLayer {
                input_ln: zeros_rms(HIDDEN, device),
                q: zeros_linear(HIDDEN, q_out, device),
                k: zeros_linear(HIDDEN, kv_out, device),
                v: zeros_linear(HIDDEN, kv_out, device),
                o: zeros_linear(q_out, HIDDEN, device),
                q_norm: zeros_rms(head_dim, device),
                k_norm: zeros_rms(head_dim, device),
                post_attn: zeros_rms(HIDDEN, device),
                pre_ff: zeros_rms(HIDDEN, device),
                gate: zeros_linear(HIDDEN, 2048, device),
                up: zeros_linear(HIDDEN, 2048, device),
                down: zeros_linear(2048, HIDDEN, device),
                post_ff: zeros_rms(HIDDEN, device),
                ple_gate: zeros_linear(HIDDEN, HIDDEN, device),
                ple_up: zeros_linear(HIDDEN, HIDDEN, device),
                ple_norm: zeros_rms(HIDDEN, device),
                scalar: Param::initialized(ParamId::new(), Tensor::<1>::ones([1], device)),
                head_dim,
                kv_heads,
                sliding,
            });
        }
        Self {
            embed: zeros_embed(VOCAB, HIDDEN, device),
            layers,
            ple: zeros_linear(HIDDEN, LAYERS * HIDDEN, device),
            ple_norm: zeros_rms(HIDDEN, device),
            norm: zeros_rms(HIDDEN, device),
            proj: zeros_linear(HIDDEN, 768, device),
            gelu: Gelu::new_approximate(),
        }
    }

    fn token_embeds(&self, ids: &[u32], device: &Device) -> Tensor<3> {
        let data: Vec<i32> = ids.iter().map(|id| *id as i32).collect();
        let tokens = Tensor::<2, Int>::from_data(TensorData::new(data, [1, ids.len()]), device);
        self.embed.forward(tokens) * (HIDDEN as f32).sqrt()
    }

    fn forward_embeds(&self, embeds: Tensor<3>) -> Result<Vec<f32>, String> {
        let [batch, seq, _] = embeds.dims();
        let device = embeds.device();
        let scale = (HIDDEN as f32).powf(-0.5);
        // The projection is layers*hidden wide. The norm weight is one layer wide, so split first.
        let projected = (self.ple.forward(embeds.clone()) * scale).reshape([batch, seq, LAYERS, HIDDEN]);
        let ple = self.ple_norm.forward(projected);
        let mut hidden = embeds;
        for (i, layer) in self.layers.iter().enumerate() {
            let slice = ple.clone().slice([0..batch, 0..seq, i..i + 1, 0..HIDDEN]).reshape([batch, seq, HIDDEN]);
            hidden = layer.forward(hidden, slice, &self.gelu);
        }
        hidden = self.proj.forward(self.norm.forward(hidden));
        let pooled = hidden.mean_dim(1).squeeze_dim::<2>(1);
        let mut vec = pooled.into_data().to_vec::<f32>().map_err(|e| e.to_string())?;
        normalize_truncate(&mut vec);
        let _ = device;
        Ok(vec)
    }
}

impl TextLayer {
    fn forward(&self, hidden: Tensor<3>, ple: Tensor<3>, gelu: &Gelu) -> Tensor<3> {
        let residual = hidden.clone();
        let x = self.input_ln.forward(hidden);
        let attn = self.attend(x);
        let mut hidden = residual + self.post_attn.forward(attn);
        let residual = hidden.clone();
        let ff = self.down.forward(gelu.forward(self.gate.forward(self.pre_ff.forward(hidden.clone()))) * self.up.forward(self.pre_ff.forward(hidden)));
        hidden = residual + self.post_ff.forward(ff);
        let residual = hidden.clone();
        let gated = gelu.forward(self.ple_gate.forward(hidden)) * ple;
        hidden = residual + self.ple_norm.forward(self.ple_up.forward(gated));
        hidden * self.scalar.val().reshape([1, 1, 1])
    }

    fn attend(&self, x: Tensor<3>) -> Tensor<3> {
        let [batch, seq, _] = x.dims();
        let hd = self.head_dim;
        let q = self.q_norm.forward(self.q.forward(x.clone()).reshape([batch, seq, HEADS, hd]));
        let k = self.k_norm.forward(self.k.forward(x.clone()).reshape([batch, seq, self.kv_heads, hd]));
        let v = rms_noscale(self.v.forward(x).reshape([batch, seq, self.kv_heads, hd]));
        let theta = if self.sliding == 1 { 10_000.0 } else { 1_000_000.0 };
        let (cos, sin) = rope_tensors(seq, hd, theta, &q.device());
        let q = apply_rope(q, cos.clone(), sin.clone());
        let k = apply_rope(k, cos, sin);
        let groups = HEADS / self.kv_heads;
        let k = repeat_kv(k, groups);
        let v = repeat_kv(v, groups);
        let scale = 1.0f32;
        let scores = q.swap_dims(1, 2).matmul(k.swap_dims(1, 2).swap_dims(2, 3)) * scale;
        let window = if self.sliding == 1 { Some(512) } else { None };
        let mask = mask_tensor(seq, window, &scores.device());
        let probs = softmax(scores + mask, 3);
        let ctx = probs.matmul(v.swap_dims(1, 2)).swap_dims(1, 2).reshape([batch, seq, HEADS * hd]);
        self.o.forward(ctx)
    }
}

impl VisionTower {
    fn new(device: &Device) -> Self {
        let mut layers = Vec::with_capacity(16);
        for _ in 0..16 {
            layers.push(VisionLayer {
                input_ln: zeros_rms(768, device),
                q: zeros_linear(768, 768, device),
                k: zeros_linear(768, 768, device),
                v: zeros_linear(768, 768, device),
                o: zeros_linear(768, 768, device),
                q_norm: zeros_rms(64, device),
                k_norm: zeros_rms(64, device),
                post_attn: zeros_rms(768, device),
                pre_ff: zeros_rms(768, device),
                gate: zeros_linear(768, 3072, device),
                up: zeros_linear(768, 3072, device),
                down: zeros_linear(3072, 768, device),
                post_ff: zeros_rms(768, device),
            });
        }
        Self { patch: zeros_linear(768, 768, device), pos: vec![0.0; 2 * 10240 * 768], layers, gelu: Gelu::new_approximate() }
    }

    fn forward(&self, patches: &[f32], grid_w: usize, grid_h: usize, device: &Device) -> Result<Tensor<3>, String> {
        let n = patches.len() / 768;
        let mut scaled = Vec::with_capacity(patches.len());
        for v in patches {
            scaled.push(2.0 * (v - 0.5));
        }
        let tokens = Tensor::<3>::from_data(TensorData::new(scaled, [1, n, 768]), device);
        let mut hidden = self.patch.forward(tokens);
        let pos = vision_positions(grid_w, grid_h, &self.pos);
        let pos_t = Tensor::<3>::from_data(TensorData::new(pos, [1, n, 768]), device);
        hidden = hidden + pos_t;
        let (cos, sin) = axial_tables(grid_w, grid_h, device);
        for layer in &self.layers {
            hidden = layer.forward(hidden, &self.gelu, cos.clone(), sin.clone());
        }
        // 3×3 average pool over the dense grid, then scale by sqrt(768).
        let pooled = pool_grid(hidden, grid_w, grid_h) * (768f32).sqrt();
        Ok(pooled)
    }
}

impl VisionLayer {
    fn forward(&self, hidden: Tensor<3>, gelu: &Gelu, cos: Tensor<4>, sin: Tensor<4>) -> Tensor<3> {
        let [batch, seq, _] = hidden.dims();
        let residual = hidden.clone();
        let x = self.input_ln.forward(hidden);
        let hd = 64;
        let heads = 12;
        let q = self.q_norm.forward(self.q.forward(x.clone()).reshape([batch, seq, heads, hd]));
        let k = self.k_norm.forward(self.k.forward(x.clone()).reshape([batch, seq, heads, hd]));
        let v = rms_noscale(self.v.forward(x).reshape([batch, seq, heads, hd]));
        let q = apply_rope(q, cos.clone(), sin.clone());
        let k = apply_rope(k, cos, sin);
        let scores = q.swap_dims(1, 2).matmul(k.swap_dims(1, 2).swap_dims(2, 3));
        let probs = softmax(scores, 3);
        let ctx = probs.matmul(v.swap_dims(1, 2)).swap_dims(1, 2).reshape([batch, seq, 768]);
        let hidden = residual + self.post_attn.forward(self.o.forward(ctx));
        let residual = hidden.clone();
        let ff_in = self.pre_ff.forward(hidden);
        let ff = self.down.forward(gelu.forward(self.gate.forward(ff_in.clone())) * self.up.forward(ff_in));
        residual + self.post_ff.forward(ff)
    }
}

fn rms_noscale<const D: usize>(x: Tensor<D>) -> Tensor<D> {
    let rms = (x.clone().square().mean_dim(D - 1) + 1e-6).sqrt();
    x / rms
}

fn rope_tensors(seq: usize, head_dim: usize, theta: f32, device: &Device) -> (Tensor<4>, Tensor<4>) {
    let mut cos = Vec::new();
    let mut sin = Vec::new();
    for p in 0..seq {
        let (c, s) = rope_cos_sin(p as f32, head_dim, theta);
        cos.extend(c);
        sin.extend(s);
    }
    let cos = Tensor::<3>::from_data(TensorData::new(cos, [1, seq, head_dim]), device).unsqueeze_dim::<4>(2);
    let sin = Tensor::<3>::from_data(TensorData::new(sin, [1, seq, head_dim]), device).unsqueeze_dim::<4>(2);
    (cos, sin)
}

fn apply_rope(x: Tensor<4>, cos: Tensor<4>, sin: Tensor<4>) -> Tensor<4> {
    let hd = x.dims()[3];
    let half = hd / 2;
    let x1 = x.clone().slice([0..x.dims()[0], 0..x.dims()[1], 0..x.dims()[2], 0..half]);
    let x2 = x.clone().slice([0..x.dims()[0], 0..x.dims()[1], 0..x.dims()[2], half..hd]);
    let rot = Tensor::cat(vec![x2.neg(), x1], 3);
    x * cos + rot * sin
}

fn repeat_kv(x: Tensor<4>, groups: usize) -> Tensor<4> {
    if groups == 1 {
        return x;
    }
    let [b, s, kv, hd] = x.dims();
    x.unsqueeze_dim::<5>(3).repeat(&[1, 1, 1, groups, 1]).reshape([b, s, kv * groups, hd])
}

fn mask_tensor(seq: usize, window: Option<usize>, device: &Device) -> Tensor<4> {
    let mut data = vec![0f32; seq * seq];
    for i in 0..seq {
        for j in 0..seq {
            if let Some(w) = window {
                if i.abs_diff(j) > w {
                    data[i * seq + j] = -1.0e4;
                }
            }
        }
    }
    Tensor::<2>::from_data(TensorData::new(data, [seq, seq]), device).unsqueeze_dim::<3>(0).unsqueeze_dim::<4>(0)
}

fn vision_positions(grid_w: usize, grid_h: usize, table: &[f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(grid_w * grid_h * 768);
    for y in 0..grid_h {
        for x in 0..grid_w {
            for d in 0..768 {
                let xe = table[x * 768 + d];
                let ye = table[(10240 + y) * 768 + d];
                out.push(xe + ye);
            }
        }
    }
    out
}

fn pool_grid(hidden: Tensor<3>, grid_w: usize, grid_h: usize) -> Tensor<3> {
    let data = hidden.into_data().to_vec::<f32>().unwrap_or_default();
    let out_w = grid_w / 3;
    let out_h = grid_h / 3;
    let mut pooled = vec![0f32; out_w * out_h * 768];
    for oy in 0..out_h {
        for ox in 0..out_w {
            for ky in 0..3 {
                for kx in 0..3 {
                    let src = ((oy * 3 + ky) * grid_w + (ox * 3 + kx)) * 768;
                    let dst = (oy * out_w + ox) * 768;
                    for d in 0..768 {
                        if src + d < data.len() {
                            pooled[dst + d] += data[src + d] / 9.0;
                        }
                    }
                }
            }
        }
    }
    let device = Device::default();
    Tensor::<3>::from_data(TensorData::new(pooled, [1, out_w * out_h, 768]), &device)
}

fn axial_tables(grid_w: usize, grid_h: usize, device: &Device) -> (Tensor<4>, Tensor<4>) {
    let mut cos = Vec::new();
    let mut sin = Vec::new();
    for y in 0..grid_h {
        for x in 0..grid_w {
            let (cy, sy) = rope_cos_sin(y as f32, 32, 100.0);
            let (cx, sx) = rope_cos_sin(x as f32, 32, 100.0);
            cos.extend(cy);
            cos.extend(cx);
            sin.extend(sy);
            sin.extend(sx);
        }
    }
    let n = (grid_w * grid_h).max(1);
    let cos = Tensor::<3>::from_data(TensorData::new(cos, [1, n, 64]), device).unsqueeze_dim::<4>(2);
    let sin = Tensor::<3>::from_data(TensorData::new(sin, [1, n, 64]), device).unsqueeze_dim::<4>(2);
    (cos, sin)
}

fn scatter_soft(embeds: Tensor<3>, soft: Tensor<3>, start: usize) -> Tensor<3> {
    let [_, seq, hidden] = embeds.dims();
    let n = soft.dims()[1].min(seq.saturating_sub(start));
    let mut host = embeds.clone().into_data().to_vec::<f32>().unwrap_or_default();
    let soft_host = soft.into_data().to_vec::<f32>().unwrap_or_default();
    for i in 0..n {
        let dst = (start + i) * hidden;
        let src = i * hidden;
        if dst + hidden <= host.len() && src + hidden <= soft_host.len() {
            host[dst..dst + hidden].copy_from_slice(&soft_host[src..src + hidden]);
        }
    }
    let device = Device::default();
    Tensor::<3>::from_data(TensorData::new(host, [1, seq, hidden]), &device)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn golden_vectors_match_when_weights_are_present() {
        let dir = match std::env::var("REMIXSEARCH_TEACHER") {
            Ok(dir) => std::path::PathBuf::from(dir),
            Err(_) => return,
        };
        let teacher = GemmaTeacher::load(Path::new(&dir)).expect("teacher");
        let a = teacher.embed_query("where are the brotli blocks stored?").unwrap();
        let b = teacher.embed_query("where are the brotli blocks stored?").unwrap();
        let mut dot = 0.0;
        for (x, y) in a.iter().zip(&b) {
            dot += x * y;
        }
        assert!(dot > 0.999, "same query cosine {dot}");
        assert_eq!(a.len(), 128);
        let fixture = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../testdata/gemma-query.txt");
        if let Ok(saved) = std::fs::read_to_string(&fixture) {
            let saved: Vec<f32> = saved.split_whitespace().filter_map(|s| s.parse().ok()).collect();
            if saved.len() == 128 {
                let mut cos = 0.0;
                for (x, y) in a.iter().zip(&saved) {
                    cos += x * y;
                }
                assert!(cos > 0.999, "golden cosine {cos}");
            }
        } else if let Some(parent) = fixture.parent() {
            let _ = std::fs::create_dir_all(parent);
            let text = a.iter().map(|v| format!("{v:.8}")).collect::<Vec<_>>().join(" ");
            let _ = std::fs::write(fixture, text);
        }
    }
}

fn normalize_truncate(v: &mut Vec<f32>) {
    let mut n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 1e-8 {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
    v.truncate(128);
    n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 1e-8 {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
}

/// Load every text and vision tensor. Split from `new` so initialization stays readable.
pub fn load_teacher_weights(teacher: &mut GemmaTeacher, dir: &Path) -> Result<(), String> {
    let store = Store::open(&dir.join("model.safetensors"))?;
    let device = teacher.device.clone();
    let embed = store.get("language_model.embed_tokens.weight")?;
    set_embed(&mut teacher.text.embed, &embed, VOCAB, HIDDEN, &device);
    let ple = store.get("language_model.ple.per_layer_model_projection.weight")?;
    set_linear(&mut teacher.text.ple, &ple, LAYERS * HIDDEN, HIDDEN, &device);
    let ple_n = store.get("language_model.ple.per_layer_projection_norm.weight")?;
    set_vec(&mut teacher.text.ple_norm.gamma, &ple_n, &device);
    let norm = store.get("language_model.norm.weight")?;
    set_vec(&mut teacher.text.norm.gamma, &norm, &device);
    let proj = store.get("language_model.embedding_projection.weight")?;
    set_linear(&mut teacher.text.proj, &proj, 768, HIDDEN, &device);
    for i in 0..LAYERS {
        let (head_dim, kv, _) = layer_kind(i);
        let prefix = format!("language_model.layers.{i}");
        let layer = &mut teacher.text.layers[i];
        set_vec(&mut layer.input_ln.gamma, &store.get(&format!("{prefix}.input_layernorm.weight"))?, &device);
        set_linear(&mut layer.q, &store.get(&format!("{prefix}.self_attn.q_proj.weight"))?, HEADS * head_dim, HIDDEN, &device);
        set_linear(&mut layer.k, &store.get(&format!("{prefix}.self_attn.k_proj.weight"))?, kv * head_dim, HIDDEN, &device);
        set_linear(&mut layer.v, &store.get(&format!("{prefix}.self_attn.v_proj.weight"))?, kv * head_dim, HIDDEN, &device);
        set_linear(&mut layer.o, &store.get(&format!("{prefix}.self_attn.o_proj.weight"))?, HIDDEN, HEADS * head_dim, &device);
        set_vec(&mut layer.q_norm.gamma, &store.get(&format!("{prefix}.self_attn.q_norm.weight"))?, &device);
        set_vec(&mut layer.k_norm.gamma, &store.get(&format!("{prefix}.self_attn.k_norm.weight"))?, &device);
        set_vec(&mut layer.post_attn.gamma, &store.get(&format!("{prefix}.post_attention_layernorm.weight"))?, &device);
        set_vec(&mut layer.pre_ff.gamma, &store.get(&format!("{prefix}.pre_feedforward_layernorm.weight"))?, &device);
        set_linear(&mut layer.gate, &store.get(&format!("{prefix}.mlp.gate_proj.weight"))?, 2048, HIDDEN, &device);
        set_linear(&mut layer.up, &store.get(&format!("{prefix}.mlp.up_proj.weight"))?, 2048, HIDDEN, &device);
        set_linear(&mut layer.down, &store.get(&format!("{prefix}.mlp.down_proj.weight"))?, HIDDEN, 2048, &device);
        set_vec(&mut layer.post_ff.gamma, &store.get(&format!("{prefix}.post_feedforward_layernorm.weight"))?, &device);
        set_linear(&mut layer.ple_gate, &store.get(&format!("{prefix}.ple_block.per_layer_input_gate.weight"))?, HIDDEN, HIDDEN, &device);
        set_linear(&mut layer.ple_up, &store.get(&format!("{prefix}.ple_block.per_layer_projection.weight"))?, HIDDEN, HIDDEN, &device);
        set_vec(&mut layer.ple_norm.gamma, &store.get(&format!("{prefix}.ple_block.post_per_layer_input_norm.weight"))?, &device);
        set_vec(&mut layer.scalar, &store.get(&format!("{prefix}.layer_scalar"))?, &device);
    }
    let patch = store.get("vision_tower.patch_embedder.input_proj.weight")?;
    set_linear(&mut teacher.vision.patch, &patch, 768, 768, &device);
    teacher.vision.pos = store.get("vision_tower.patch_embedder.position_embedding_table")?;
    for i in 0..16 {
        let prefix = format!("vision_tower.encoder.layers.{i}");
        let layer = &mut teacher.vision.layers[i];
        let lin = |name: &str| store.get(&format!("{prefix}.{name}.linear.weight"));
        set_vec(&mut layer.input_ln.gamma, &store.get(&format!("{prefix}.input_layernorm.weight"))?, &device);
        set_linear(&mut layer.q, &lin("self_attn.q_proj")?, 768, 768, &device);
        set_linear(&mut layer.k, &lin("self_attn.k_proj")?, 768, 768, &device);
        set_linear(&mut layer.v, &lin("self_attn.v_proj")?, 768, 768, &device);
        set_linear(&mut layer.o, &lin("self_attn.o_proj")?, 768, 768, &device);
        set_vec(&mut layer.q_norm.gamma, &store.get(&format!("{prefix}.self_attn.q_norm.weight"))?, &device);
        set_vec(&mut layer.k_norm.gamma, &store.get(&format!("{prefix}.self_attn.k_norm.weight"))?, &device);
        set_vec(&mut layer.post_attn.gamma, &store.get(&format!("{prefix}.post_attention_layernorm.weight"))?, &device);
        set_vec(&mut layer.pre_ff.gamma, &store.get(&format!("{prefix}.pre_feedforward_layernorm.weight"))?, &device);
        set_linear(&mut layer.gate, &lin("mlp.gate_proj")?, 3072, 768, &device);
        set_linear(&mut layer.up, &lin("mlp.up_proj")?, 3072, 768, &device);
        set_linear(&mut layer.down, &lin("mlp.down_proj")?, 768, 3072, &device);
        set_vec(&mut layer.post_ff.gamma, &store.get(&format!("{prefix}.post_feedforward_layernorm.weight"))?, &device);
    }
    let vision_proj = store.get("embed_vision.embedding_projection.weight")?;
    set_linear(&mut teacher.embed_vision, &vision_proj, HIDDEN, 768, &device);
    Ok(())
}
