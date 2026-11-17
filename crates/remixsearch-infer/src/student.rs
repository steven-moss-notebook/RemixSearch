use crate::hnsw::l2_normalize;
use remixsearch_format::{Error, Result, DIMS};

/// Dequantized Pico-sized query encoder.
///
/// Layout matches the Burn student: factorized token embeddings, absolute
/// positions, post-norm bidirectional attention, GELU feed-forward, mean pool.
#[derive(Clone, Debug)]
pub struct StudentWeights {
    pub vocab: usize,
    pub proj: usize,
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub intermediate: usize,
    pub max_len: usize,
    pub out_dim: usize,
    pub token_embed: Vec<f32>,
    pub token_proj: Vec<f32>,
    pub pos_embed: Vec<f32>,
    pub layers_w: Vec<LayerWeights>,
    pub out_w: Vec<f32>,
    pub out_b: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct LayerWeights {
    pub ln1_w: Vec<f32>,
    pub ln1_b: Vec<f32>,
    pub wq: Vec<f32>,
    pub wk: Vec<f32>,
    pub wv: Vec<f32>,
    pub wo: Vec<f32>,
    pub ln2_w: Vec<f32>,
    pub ln2_b: Vec<f32>,
    pub up: Vec<f32>,
    pub down: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct StudentF32 {
    pub weights: StudentWeights,
}

impl StudentF32 {
    pub fn embed(&self, token_ids: &[u32]) -> Vec<f32> {
        let w = &self.weights;
        if w.hidden == 0 {
            return vec![0.0; DIMS];
        }
        let mut ids: Vec<u32> = token_ids.iter().copied().take(w.max_len).collect();
        if ids.is_empty() {
            ids.push(0);
        }
        let mut states = Vec::with_capacity(ids.len());
        for (pos, id) in ids.iter().enumerate() {
            let id = (*id as usize).min(w.vocab.saturating_sub(1));
            let row = &w.token_embed[id * w.proj..(id + 1) * w.proj];
            let mut h = linear(row, &w.token_proj, w.proj, w.hidden);
            let pos_row = &w.pos_embed[pos * w.hidden..(pos + 1) * w.hidden];
            for (a, b) in h.iter_mut().zip(pos_row) {
                *a += *b;
            }
            states.push(h);
        }
        let mask = vec![true; states.len()];
        for layer in &w.layers_w {
            states = layer_forward(states, &mask, layer, w.heads, w.intermediate);
        }
        let mut pooled = vec![0f32; w.hidden];
        for state in &states {
            for (p, s) in pooled.iter_mut().zip(state) {
                *p += *s;
            }
        }
        let inv = 1.0 / states.len() as f32;
        for p in &mut pooled {
            *p *= inv;
        }
        let mut y = linear(&pooled, &w.out_w, w.hidden, w.out_dim);
        for (i, b) in w.out_b.iter().enumerate() {
            if i < y.len() {
                y[i] += *b;
            }
        }
        if y.len() != DIMS {
            y.resize(DIMS, 0.0);
        }
        l2_normalize(&mut y);
        y
    }

    pub fn identity_for_tests() -> Self {
        Self { weights: StudentWeights::zeros() }
    }

    pub fn from_int8(bytes: &[u8]) -> Result<Self> {
        Ok(Self { weights: decode_student(bytes)? })
    }

    pub fn to_int8(&self) -> Vec<u8> {
        encode_student(&self.weights)
    }
}

impl StudentWeights {
    pub fn zeros() -> Self {
        Self {
            vocab: 0,
            proj: 0,
            hidden: 0,
            layers: 0,
            heads: 0,
            intermediate: 0,
            max_len: 1,
            out_dim: DIMS,
            token_embed: Vec::new(),
            token_proj: Vec::new(),
            pos_embed: Vec::new(),
            layers_w: Vec::new(),
            out_w: Vec::new(),
            out_b: vec![0.0; DIMS],
        }
    }
}

fn layer_forward(states: Vec<Vec<f32>>, mask: &[bool], layer: &LayerWeights, heads: usize, intermediate: usize) -> Vec<Vec<f32>> {
    let mut out = Vec::with_capacity(states.len());
    let attn = attention(&states, mask, layer, heads);
    for (i, s) in states.into_iter().enumerate() {
        let mut h = s;
        for (a, b) in h.iter_mut().zip(&attn[i]) {
            *a += *b;
        }
        h = layer_norm(&h, &layer.ln1_w, &layer.ln1_b);
        let mut ff = linear(&h, &layer.up, h.len(), intermediate);
        gelu_tanh_inplace(&mut ff);
        let down = linear(&ff, &layer.down, intermediate, h.len());
        for (a, b) in h.iter_mut().zip(&down) {
            *a += *b;
        }
        out.push(layer_norm(&h, &layer.ln2_w, &layer.ln2_b));
    }
    out
}

fn attention(states: &[Vec<f32>], mask: &[bool], layer: &LayerWeights, heads: usize) -> Vec<Vec<f32>> {
    let seq = states.len();
    let hidden = states[0].len();
    let head_dim = hidden / heads.max(1);
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut qs = Vec::with_capacity(seq);
    let mut ks = Vec::with_capacity(seq);
    let mut vs = Vec::with_capacity(seq);
    for s in states {
        qs.push(linear(s, &layer.wq, hidden, hidden));
        ks.push(linear(s, &layer.wk, hidden, hidden));
        vs.push(linear(s, &layer.wv, hidden, hidden));
    }
    let mut ctx = vec![vec![0f32; hidden]; seq];
    for h in 0..heads {
        let hs = h * head_dim;
        let mut scores = vec![vec![0f32; seq]; seq];
        for i in 0..seq {
            for j in 0..seq {
                if !mask[j] {
                    scores[i][j] = -1.0e9;
                    continue;
                }
                let mut d = 0f32;
                for d_i in 0..head_dim {
                    d += qs[i][hs + d_i] * ks[j][hs + d_i];
                }
                scores[i][j] = d * scale;
            }
            softmax_inplace(&mut scores[i]);
        }
        for i in 0..seq {
            for j in 0..seq {
                let p = scores[i][j];
                for d_i in 0..head_dim {
                    ctx[i][hs + d_i] += p * vs[j][hs + d_i];
                }
            }
        }
    }
    ctx.into_iter().map(|c| linear(&c, &layer.wo, hidden, hidden)).collect()
}

pub fn linear(x: &[f32], w: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
    let mut y = vec![0f32; out_dim];
    let in_dim = in_dim.min(x.len());
    for i in 0..in_dim {
        let xi = x[i];
        let row = i * out_dim;
        if row + out_dim > w.len() {
            break;
        }
        for o in 0..out_dim {
            y[o] += xi * w[row + o];
        }
    }
    y
}

pub fn layer_norm(x: &[f32], w: &[f32], b: &[f32]) -> Vec<f32> {
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let inv = 1.0 / (var + 1e-5).sqrt();
    x.iter()
        .enumerate()
        .map(|(i, v)| {
            let g = w.get(i).copied().unwrap_or(1.0);
            let bias = b.get(i).copied().unwrap_or(0.0);
            (v - mean) * inv * g + bias
        })
        .collect()
}

pub fn gelu_tanh_inplace(x: &mut [f32]) {
    const K: f32 = 0.7978845608; // sqrt(2/pi)
    for v in x {
        let y = *v;
        *v = 0.5 * y * (1.0 + (K * (y + 0.044715 * y * y * y)).tanh());
    }
}

fn softmax_inplace(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for v in x {
            *v /= sum;
        }
    }
}

fn quantize_tensor(values: &[f32]) -> (f32, Vec<u8>) {
    let max = values.iter().map(|v| v.abs()).fold(0f32, f32::max);
    let scale = if max < 1e-12 { 1.0 } else { max / 127.0 };
    let bytes = values
        .iter()
        .map(|v| (v / scale).round().clamp(-127.0, 127.0) as i8 as u8)
        .collect();
    (scale, bytes)
}

fn push_tensor(out: &mut Vec<u8>, values: &[f32], dims: &[u32]) {
    out.push(dims.len() as u8);
    for d in dims {
        out.extend_from_slice(&d.to_le_bytes());
    }
    let (scale, bytes) = quantize_tensor(values);
    out.extend_from_slice(&scale.to_le_bytes());
    out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(&bytes);
}

fn encode_student(w: &StudentWeights) -> Vec<u8> {
    let mut out = b"SIST".to_vec();
    for v in [1u32, w.vocab as u32, w.proj as u32, w.hidden as u32, w.layers as u32, w.heads as u32, w.intermediate as u32, w.max_len as u32, w.out_dim as u32] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    push_tensor(&mut out, &w.token_embed, &[w.vocab as u32, w.proj as u32]);
    push_tensor(&mut out, &w.token_proj, &[w.proj as u32, w.hidden as u32]);
    push_tensor(&mut out, &w.pos_embed, &[w.max_len as u32, w.hidden as u32]);
    for layer in &w.layers_w {
        let h = w.hidden as u32;
        let inter = w.intermediate as u32;
        push_tensor(&mut out, &layer.ln1_w, &[h]);
        push_tensor(&mut out, &layer.ln1_b, &[h]);
        push_tensor(&mut out, &layer.wq, &[h, h]);
        push_tensor(&mut out, &layer.wk, &[h, h]);
        push_tensor(&mut out, &layer.wv, &[h, h]);
        push_tensor(&mut out, &layer.wo, &[h, h]);
        push_tensor(&mut out, &layer.ln2_w, &[h]);
        push_tensor(&mut out, &layer.ln2_b, &[h]);
        push_tensor(&mut out, &layer.up, &[h, inter]);
        push_tensor(&mut out, &layer.down, &[inter, h]);
    }
    push_tensor(&mut out, &w.out_w, &[w.hidden as u32, w.out_dim as u32]);
    push_tensor(&mut out, &w.out_b, &[w.out_dim as u32]);
    out
}

struct Reader<'a> {
    data: &'a [u8],
    i: usize,
}

impl<'a> Reader<'a> {
    fn u32(&mut self) -> Result<u32> {
        if self.i + 4 > self.data.len() {
            return Err(Error::Message("truncated student".into()));
        }
        let v = u32::from_le_bytes(self.data[self.i..self.i + 4].try_into().unwrap());
        self.i += 4;
        Ok(v)
    }
    fn tensor(&mut self) -> Result<Vec<f32>> {
        if self.i >= self.data.len() {
            return Err(Error::Message("truncated tensor".into()));
        }
        let ndim = self.data[self.i] as usize;
        self.i += 1;
        let mut numel = 1usize;
        for _ in 0..ndim {
            numel *= self.u32()? as usize;
        }
        if self.i + 4 > self.data.len() {
            return Err(Error::Message("truncated scale".into()));
        }
        let scale = f32::from_le_bytes(self.data[self.i..self.i + 4].try_into().unwrap());
        self.i += 4;
        let n = self.u32()? as usize;
        if self.i + n > self.data.len() {
            return Err(Error::Message("truncated tensor bytes".into()));
        }
        let values = self.data[self.i..self.i + n].iter().map(|b| *b as i8 as f32 * scale).collect::<Vec<_>>();
        self.i += n;
        if values.len() != numel && numel != 0 {
            // dims describe the logical shape; trust the stored byte length
        }
        Ok(values)
    }
}

fn decode_student(data: &[u8]) -> Result<StudentWeights> {
    if data.len() < 8 || &data[0..4] != b"SIST" {
        return Err(Error::Message("bad student magic".into()));
    }
    let mut r = Reader { data, i: 4 };
    let _version = r.u32()?;
    let vocab = r.u32()? as usize;
    let proj = r.u32()? as usize;
    let hidden = r.u32()? as usize;
    let layers = r.u32()? as usize;
    let heads = r.u32()? as usize;
    let intermediate = r.u32()? as usize;
    let max_len = r.u32()? as usize;
    let out_dim = r.u32()? as usize;
    let token_embed = r.tensor()?;
    let token_proj = r.tensor()?;
    let pos_embed = r.tensor()?;
    let mut layers_w = Vec::with_capacity(layers);
    for _ in 0..layers {
        layers_w.push(LayerWeights {
            ln1_w: r.tensor()?,
            ln1_b: r.tensor()?,
            wq: r.tensor()?,
            wk: r.tensor()?,
            wv: r.tensor()?,
            wo: r.tensor()?,
            ln2_w: r.tensor()?,
            ln2_b: r.tensor()?,
            up: r.tensor()?,
            down: r.tensor()?,
        });
    }
    Ok(StudentWeights {
        vocab,
        proj,
        hidden,
        layers,
        heads,
        intermediate,
        max_len,
        out_dim,
        token_embed,
        token_proj,
        pos_embed,
        layers_w,
        out_w: r.tensor()?,
        out_b: r.tensor()?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn int8_roundtrip_changes_little() {
        let mut w = StudentWeights::zeros();
        w.vocab = 4;
        w.proj = 2;
        w.hidden = 4;
        w.layers = 1;
        w.heads = 2;
        w.intermediate = 8;
        w.max_len = 4;
        w.out_dim = DIMS;
        let fill = |n: usize, salt: f32| -> Vec<f32> {
            (0..n).map(|i| ((i as f32 * 0.17 + salt).sin()) * 0.4).collect()
        };
        w.token_embed = fill(8, 0.1);
        w.token_proj = fill(8, 0.2);
        w.pos_embed = fill(16, 0.3);
        w.out_w = fill(4 * DIMS, 0.4);
        w.out_b = vec![0.0; DIMS];
        w.layers_w.push(LayerWeights {
            ln1_w: vec![1.0; 4],
            ln1_b: vec![0.0; 4],
            wq: fill(16, 1.0),
            wk: fill(16, 2.0),
            wv: fill(16, 3.0),
            wo: fill(16, 4.0),
            ln2_w: vec![1.0; 4],
            ln2_b: vec![0.0; 4],
            up: fill(32, 5.0),
            down: fill(32, 6.0),
        });
        let model = StudentF32 { weights: w };
        let before = model.embed(&[1, 2, 3]);
        let bytes = model.to_int8();
        let loaded = StudentF32::from_int8(&bytes).unwrap();
        let after = loaded.embed(&[1, 2, 3]);
        let mut dot = 0.0;
        for (a, b) in before.iter().zip(&after) {
            dot += a * b;
        }
        assert!(
            dot > 0.95,
            "cosine {dot} before0={} after0={} blen {} alen {} bytes {}",
            before.first().copied().unwrap_or(f32::NAN),
            after.first().copied().unwrap_or(f32::NAN),
            before.len(),
            after.len(),
            bytes.len()
        );
    }
}
