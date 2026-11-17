//! Pico-sized query encoder. The forward matches `remixsearch_infer::StudentF32`:
//! factorized embeddings, absolute positions, post-norm attention, GELU MLP, mean pool.

use burn::nn::{
    Embedding, EmbeddingConfig, Gelu, Initializer, LayerNorm, LayerNormConfig, Linear, LinearConfig,
};
use burn::prelude::*;
use burn::tensor::activation::softmax;
use remixsearch_infer::{LayerWeights, StudentF32, StudentWeights};

#[derive(Config, Debug)]
pub struct QueryEncoderConfig {
    #[config(default = 8192)]
    pub vocab: usize,
    #[config(default = 16)]
    pub proj: usize,
    #[config(default = 128)]
    pub hidden: usize,
    #[config(default = 2)]
    pub layers: usize,
    #[config(default = 4)]
    pub heads: usize,
    #[config(default = 512)]
    pub intermediate: usize,
    #[config(default = 128)]
    pub max_len: usize,
    #[config(default = 128)]
    pub out_dim: usize,
}

#[derive(Module, Debug)]
pub struct EncLayer {
    ln1: LayerNorm,
    wq: Linear,
    wk: Linear,
    wv: Linear,
    wo: Linear,
    ln2: LayerNorm,
    up: Linear,
    down: Linear,
    gelu: Gelu,
    heads: usize,
}

#[derive(Module, Debug)]
pub struct QueryEncoder {
    pub token_embed: Embedding,
    pub token_proj: Linear,
    pub pos_embed: Embedding,
    pub layers: Vec<EncLayer>,
    pub out: Linear,
    pub hidden: usize,
    pub max_len: usize,
}

impl QueryEncoderConfig {
    pub fn init(&self, device: &Device) -> QueryEncoder {
        let init = Initializer::Normal { mean: 0.0, std: 0.02 };
        let mut layers = Vec::with_capacity(self.layers);
        for _ in 0..self.layers {
            layers.push(EncLayer {
                ln1: LayerNormConfig::new(self.hidden).with_epsilon(1e-5).init(device),
                wq: LinearConfig::new(self.hidden, self.hidden).with_bias(false).with_initializer(init.clone()).init(device),
                wk: LinearConfig::new(self.hidden, self.hidden).with_bias(false).with_initializer(init.clone()).init(device),
                wv: LinearConfig::new(self.hidden, self.hidden).with_bias(false).with_initializer(init.clone()).init(device),
                wo: LinearConfig::new(self.hidden, self.hidden).with_bias(false).with_initializer(init.clone()).init(device),
                ln2: LayerNormConfig::new(self.hidden).with_epsilon(1e-5).init(device),
                up: LinearConfig::new(self.hidden, self.intermediate).with_bias(false).with_initializer(init.clone()).init(device),
                down: LinearConfig::new(self.intermediate, self.hidden).with_bias(false).with_initializer(init.clone()).init(device),
                gelu: Gelu::new_approximate(),
                heads: self.heads,
            });
        }
        QueryEncoder {
            token_embed: EmbeddingConfig::new(self.vocab, self.proj).with_initializer(init.clone()).init(device),
            token_proj: LinearConfig::new(self.proj, self.hidden).with_bias(false).with_initializer(init.clone()).init(device),
            pos_embed: EmbeddingConfig::new(self.max_len, self.hidden).with_initializer(init).init(device),
            layers,
            out: LinearConfig::new(self.hidden, self.out_dim).init(device),
            hidden: self.hidden,
            max_len: self.max_len,
        }
    }
}

impl QueryEncoder {
    pub fn embed(&self, tokens: Tensor<2, Int>, mask: Tensor<2>) -> Tensor<2> {
        let [batch, seq] = tokens.dims();
        let device = &tokens.device();
        let positions = Tensor::<1, Int>::arange(0..seq as i64, device)
            .reshape([1, seq])
            .repeat(&[batch, 1]);
        let mut x = self.token_proj.forward(self.token_embed.forward(tokens)) + self.pos_embed.forward(positions);
        for layer in &self.layers {
            x = layer.forward(x, mask.clone());
        }
        let mask = mask.clone().unsqueeze_dim::<3>(2);
        let summed = (x.clone() * mask.clone()).sum_dim(1);
        let denom = mask.sum_dim(1).clamp_min(1.0);
        let pooled = summed / denom;
        let y = self.out.forward(pooled.squeeze_dim::<2>(1));
        l2_normalize(y)
    }

    pub fn export_weights(&self) -> StudentWeights {
        let token_embed = tensor2(&self.token_embed.weight.val());
        let [vocab, proj] = self.token_embed.weight.val().dims();
        let token_proj = tensor2(&self.token_proj.weight.val());
        let pos_embed = tensor2(&self.pos_embed.weight.val());
        let [max_len, hidden] = self.pos_embed.weight.val().dims();
        let out_w = tensor2(&self.out.weight.val());
        let out_dim = self.out.weight.val().dims()[1];
        let out_b = self
            .out
            .bias
            .as_ref()
            .map(|b| tensor1(&b.val()))
            .unwrap_or_else(|| vec![0.0; out_dim]);
        let mut layers_w = Vec::new();
        let mut intermediate = 0;
        let mut heads = 4;
        for layer in &self.layers {
            heads = layer.heads;
            intermediate = layer.up.weight.val().dims()[1];
            layers_w.push(LayerWeights {
                ln1_w: tensor1(&layer.ln1.gamma.val()),
                ln1_b: layer.ln1.beta.as_ref().map(|b| tensor1(&b.val())).unwrap_or_else(|| vec![0.0; hidden]),
                wq: tensor2(&layer.wq.weight.val()),
                wk: tensor2(&layer.wk.weight.val()),
                wv: tensor2(&layer.wv.weight.val()),
                wo: tensor2(&layer.wo.weight.val()),
                ln2_w: tensor1(&layer.ln2.gamma.val()),
                ln2_b: layer.ln2.beta.as_ref().map(|b| tensor1(&b.val())).unwrap_or_else(|| vec![0.0; hidden]),
                up: tensor2(&layer.up.weight.val()),
                down: tensor2(&layer.down.weight.val()),
            });
        }
        let weights = StudentWeights {
            vocab,
            proj,
            hidden,
            layers: layers_w.len(),
            heads,
            intermediate,
            max_len,
            out_dim: out_b.len(),
            token_embed,
            token_proj,
            pos_embed,
            layers_w,
            out_w,
            out_b,
        };
        let _ = StudentF32 { weights: weights.clone() };
        weights
    }
}

impl EncLayer {
    fn forward(&self, x: Tensor<3>, mask: Tensor<2>) -> Tensor<3> {
        let h = self.ln1.forward(x.clone() + self.attention(x.clone(), mask));
        let ff = self.down.forward(self.gelu.forward(self.up.forward(h.clone())));
        self.ln2.forward(h + ff)
    }

    fn attention(&self, x: Tensor<3>, mask: Tensor<2>) -> Tensor<3> {
        let [batch, seq, hidden] = x.dims();
        let heads = self.heads.max(1);
        let hd = hidden / heads;
        let scale = 1.0 / (hd as f32).sqrt();
        let q = self.wq.forward(x.clone()).reshape([batch, seq, heads, hd]).swap_dims(1, 2);
        let k = self.wk.forward(x.clone()).reshape([batch, seq, heads, hd]).swap_dims(1, 2);
        let v = self.wv.forward(x).reshape([batch, seq, heads, hd]).swap_dims(1, 2);
        let scores = q.matmul(k.swap_dims(2, 3)) * scale;
        let blocked = (mask.neg() + 1.0) * -1.0e4;
        let blocked = blocked.unsqueeze_dim::<3>(1).unsqueeze_dim::<4>(1);
        let probs = softmax(scores + blocked, 3);
        let ctx = probs.matmul(v).swap_dims(1, 2).reshape([batch, seq, hidden]);
        self.wo.forward(ctx)
    }
}

fn l2_normalize(y: Tensor<2>) -> Tensor<2> {
    let denom = y.clone().square().sum_dim(1).sqrt().clamp_min(1e-8);
    y / denom
}

fn tensor1(t: &Tensor<1>) -> Vec<f32> {
    t.clone().into_data().to_vec::<f32>().expect("f32")
}

fn tensor2(t: &Tensor<2>) -> Vec<f32> {
    t.clone().into_data().to_vec::<f32>().expect("f32")
}
