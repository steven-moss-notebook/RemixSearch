//! Distill a frozen teacher into the query encoder with Burn's learner.

use crate::student::{QueryEncoder, QueryEncoderConfig};
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataset::InMemDataset;
use burn::grad_clipping::GradientClippingConfig;
use burn::lr_scheduler::cosine::CosineAnnealingLrSchedulerConfig;
use burn::lr_scheduler::linear::LinearLrSchedulerConfig;
use burn::lr_scheduler::sequential::SequentialLrSchedulerConfig;
use burn::lr_scheduler::module_lr_scheduler::ModuleLrScheduler;
use burn::lr_scheduler::LrSchedulerConfig;
use burn::optim::AdamWConfig;
use burn::prelude::*;
use burn::tensor::activation::{log_softmax, softmax};
use burn::tensor::Distribution;
use burn::train::metric::state::{FormatOptions, NumericMetricState};
use burn::train::metric::{
    Adaptor, Metric, MetricAttributes, MetricMetadata, MetricName, Numeric, NumericAttributes, NumericEntry,
    SerializedEntry,
};
use burn::train::{
    ExecutionStrategy, InferenceStep, Learner, RegressionOutput, SupervisedTraining, TrainOutput, TrainStep,
};
use remixsearch_build::Chunk;
use remixsearch_format::{QueryRecord, DIMS};
use remixsearch_infer::{l2_normalize, CompactBpe, StudentWeights};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

#[derive(Clone, Debug)]
pub struct DistillConfig {
    pub epochs: usize,
    pub batch_size: usize,
    pub max_pairs: usize,
    pub lr: f64,
    pub vocab: usize,
    pub artifact_dir: PathBuf,
}

/// Contrastive batches start here. Smaller sets do not regularize a recommender loss.
pub const MIN_CONTRAST: usize = 2048;
/// Softmax temperature for the contrastive term. 0.07 matches the usual CLIP scale.
const TEMPERATURE: f32 = 0.07;
/// Noise on negative cosine scores. Small enough that a real margin survives the anneal.
const CONTRAST_NOISE: f64 = 0.05;

impl Default for DistillConfig {
    fn default() -> Self {
        Self {
            epochs: 80,
            batch_size: MIN_CONTRAST,
            max_pairs: 4096,
            lr: 1e-3,
            vocab: 8192,
            artifact_dir: PathBuf::from("artifacts"),
        }
    }
}

#[derive(Clone, Debug)]
struct Pair {
    tokens: Vec<i32>,
    teacher: Vec<f32>,
    positive: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct Batch {
    tokens: Tensor<2, Int>,
    mask: Tensor<2>,
    teacher: Tensor<2>,
    positive: Tensor<2>,
    /// Fresh draw of corpus vectors. Column 0 of the contrastive scores is `positive`;
    /// these are the randomized denominator.
    negatives: Tensor<2>,
}

struct PairBatcher {
    max_len: usize,
    pad_id: i32,
    corpus: Arc<Vec<f32>>,
    n_corpus: usize,
    contrast_size: usize,
    rng: Arc<Mutex<Lcg>>,
}

impl Clone for PairBatcher {
    fn clone(&self) -> Self {
        Self {
            max_len: self.max_len,
            pad_id: self.pad_id,
            corpus: self.corpus.clone(),
            n_corpus: self.n_corpus,
            contrast_size: self.contrast_size,
            rng: self.rng.clone(),
        }
    }
}

/// Numeric state for a contrastive draw. Held in a mutex so every batch, including
/// batches built on a loader thread, advances the same stream.
struct Lcg(u64);

impl Lcg {
    fn next_index(&mut self, n: usize) -> usize {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.0 >> 33) as usize % n.max(1)
    }
}

impl PairBatcher {
    fn sample_negatives(&self) -> Vec<f32> {
        let mut rng = self.rng.lock().unwrap_or_else(|e| e.into_inner());
        let mut out = vec![0f32; self.contrast_size * DIMS];
        if self.n_corpus == 0 {
            return out;
        }
        for i in 0..self.contrast_size {
            let idx = rng.next_index(self.n_corpus);
            let src = idx * DIMS;
            out[i * DIMS..(i + 1) * DIMS].copy_from_slice(&self.corpus[src..src + DIMS]);
        }
        out
    }
}

impl Batcher<Pair, Batch> for PairBatcher {
    fn batch(&self, items: Vec<Pair>, device: &Device) -> Batch {
        let batch = items.len();
        let seq = items.iter().map(|i| i.tokens.len()).max().unwrap_or(1).min(self.max_len).max(1);
        let mut tokens = vec![self.pad_id; batch * seq];
        let mut mask = vec![0f32; batch * seq];
        let mut teacher = vec![0f32; batch * DIMS];
        let mut positive = vec![0f32; batch * DIMS];
        for (b, item) in items.iter().enumerate() {
            for (s, id) in item.tokens.iter().take(seq).enumerate() {
                tokens[b * seq + s] = *id;
                mask[b * seq + s] = 1.0;
            }
            teacher[b * DIMS..(b + 1) * DIMS].copy_from_slice(&item.teacher);
            positive[b * DIMS..(b + 1) * DIMS].copy_from_slice(&item.positive);
        }
        let negatives = self.sample_negatives();
        let n_neg = self.contrast_size.max(1);
        Batch {
            tokens: Tensor::<2, Int>::from_data(TensorData::new(tokens, [batch, seq]), device),
            mask: Tensor::<2>::from_data(TensorData::new(mask, [batch, seq]), device),
            teacher: Tensor::<2>::from_data(TensorData::new(teacher, [batch, DIMS]), device),
            positive: Tensor::<2>::from_data(TensorData::new(positive, [batch, DIMS]), device),
            negatives: Tensor::<2>::from_data(TensorData::new(negatives, [n_neg, DIMS]), device),
        }
    }
}

trait Scalar: Clone + Send + Sync + 'static {
    fn tensor(&self) -> &Tensor<1>;
}

#[derive(Clone)]
struct CosineIn { tensor: Tensor<1> }
#[derive(Clone)]
struct RecallIn { tensor: Tensor<1> }

impl Scalar for CosineIn { fn tensor(&self) -> &Tensor<1> { &self.tensor } }
impl Scalar for RecallIn { fn tensor(&self) -> &Tensor<1> { &self.tensor } }

struct ScalarMetric<T> {
    name: Arc<String>,
    state: NumericMetricState,
    _tag: std::marker::PhantomData<T>,
}

impl<T> Clone for ScalarMetric<T> {
    fn clone(&self) -> Self {
        Self { name: self.name.clone(), state: self.state.clone(), _tag: std::marker::PhantomData }
    }
}

impl<T> ScalarMetric<T> {
    fn new(name: &str) -> Self {
        Self { name: Arc::new(name.into()), state: NumericMetricState::default(), _tag: std::marker::PhantomData }
    }
}

impl<T: Scalar> Metric for ScalarMetric<T> {
    type Input = T;

    fn update(&mut self, item: &Self::Input, _: &MetricMetadata) -> Result<SerializedEntry, burn::tensor::TensorReadError> {
        let [n] = item.tensor().dims();
        let value = item.tensor().clone().mean().try_into_scalar::<f64>()?;
        self.state.update(value, n.max(1));
        Ok(self.state.compute_update(FormatOptions::new(self.name()).precision(3)))
    }
    fn compute(&mut self) -> Result<SerializedEntry, burn::tensor::TensorReadError> {
        Ok(self.state.compute_final(FormatOptions::new(self.name()).precision(3)))
    }
    fn clear(&mut self) { self.state.reset(); }
    fn name(&self) -> MetricName { self.name.clone() }
    fn attributes(&self) -> MetricAttributes {
        NumericAttributes { unit: None, higher_is_better: true }.into()
    }
}

impl<T: Scalar> Numeric for ScalarMetric<T> {
    fn value(&self) -> Option<NumericEntry> { Some(self.state.current_value()) }
    fn running_value(&self) -> Option<NumericEntry> { Some(self.state.running_value()) }
    fn final_value(&self) -> NumericEntry { self.state.final_value() }
}

impl Adaptor<CosineIn> for RegressionOutput {
    fn adapt(&self) -> CosineIn {
        CosineIn { tensor: self.output.clone().slice([0..1, 0..1]).reshape([1]) }
    }
}

impl Adaptor<RecallIn> for RegressionOutput {
    fn adapt(&self) -> RecallIn {
        RecallIn { tensor: self.output.clone().slice([0..1, 1..2]).reshape([1]) }
    }
}

fn loss_on(model: &QueryEncoder, batch: Batch, regularize: bool) -> RegressionOutput {
    let q_s = model.embed(batch.tokens, batch.mask.clone());
    let q_t = batch.teacher;
    let cosine = (q_s.clone() * q_t.clone()).sum_dim(1).squeeze_dim::<1>(1);
    let align = (cosine.clone().neg() + 1.0).mean();
    let scores_s = contrastive_scores(q_s, batch.positive.clone(), batch.negatives.clone(), regularize);
    let scores_t = contrastive_scores(q_t, batch.positive, batch.negatives, false);
    let log_p_s = log_softmax(scores_s.clone() / TEMPERATURE, 1);
    let p_t = softmax(scores_t / TEMPERATURE, 1);
    let kl = (p_t.clone() * (p_t.log() - log_p_s.clone())).sum_dim(1).mean();
    let rows = log_p_s.dims()[0];
    let nce = log_p_s.slice([0..rows, 0..1]).neg().mean();
    let loss = (align * 0.2 + kl * 0.4 + nce * 0.4).reshape([1]);
    let hits = scores_s.argmax(1).equal_elem(0).float().mean().reshape([1, 1]);
    let cosine = cosine.mean().reshape([1, 1]);
    let stats = Tensor::cat(vec![cosine, hits], 1);
    RegressionOutput::new(loss, stats.clone(), stats)
}

/// Positive score in column 0, then a shared random document sample.
/// Training adds Gaussian noise to the negative columns only, so the
/// contrastive denominator jitters without moving the labeled page.
fn contrastive_scores(q: Tensor<2>, positive: Tensor<2>, negatives: Tensor<2>, regularize: bool) -> Tensor<2> {
    let pos = (q.clone() * positive).sum_dim(1);
    let mut neg = q.matmul(negatives.transpose());
    if regularize {
        let noise = Tensor::<2>::random(neg.dims(), Distribution::Normal(0.0, CONTRAST_NOISE), &neg.device());
        neg = neg + noise;
    }
    Tensor::cat(vec![pos, neg], 1)
}

impl TrainStep for QueryEncoder {
    type Input = Batch;
    type Output = RegressionOutput;

    fn step(&self, item: Batch) -> TrainOutput<RegressionOutput> {
        let out = loss_on(self, item, true);
        let grads = out.loss.backward();
        TrainOutput::new(self, grads, out)
    }
}

impl InferenceStep for QueryEncoder {
    type Input = Batch;
    type Output = RegressionOutput;

    fn step(&self, item: Batch) -> RegressionOutput {
        loss_on(self, item, false)
    }
}

fn build_pairs(
    chunks: &[Chunk],
    vectors: &[Vec<f32>],
    queries: &[QueryRecord],
    tokenizer: &CompactBpe,
    embed_query: &dyn Fn(&str) -> Vec<f32>,
    max_pairs: usize,
) -> Vec<Pair> {
    let mut index: Vec<(String, usize)> = Vec::new();
    for (i, chunk) in chunks.iter().enumerate() {
        index.push((chunk.url.clone(), i));
        for image in &chunk.images {
            index.push((image.src.clone(), i));
        }
    }
    let mut pairs = Vec::new();
    for query in select_queries(queries, max_pairs) {
        if query.unanswerable || query.held_out {
            continue;
        }
        let Some((_, pos)) = index.iter().find(|(url, _)| url == &query.positive_url) else { continue };
        let mut q = embed_query(&query.query);
        if dot(&q, &vectors[*pos]) < 0.35 {
            // The teacher did not put this question near its answer. Train the
            // query encoder onto the answer vector that is stored in the index.
            q = vectors[*pos].clone();
        }
        let positive = vectors[*pos].clone();
        let mut tokens: Vec<i32> = tokenizer.encode(&query.query).into_iter().map(|id| id as i32).collect();
        if tokens.is_empty() {
            tokens.push(0);
        }
        pairs.push(Pair { tokens, teacher: q, positive });
        if pairs.len() >= max_pairs {
            break;
        }
    }
    pairs
}

fn select_queries(queries: &[QueryRecord], max_pairs: usize) -> Vec<&QueryRecord> {
    let mut buckets: std::collections::BTreeMap<String, Vec<&QueryRecord>> = std::collections::BTreeMap::new();
    for query in queries {
        if query.unanswerable || query.held_out {
            continue;
        }
        let subject = subject_of(&query.positive_url);
        buckets.entry(subject).or_default().push(query);
    }
    for bucket in buckets.values_mut() {
        bucket.sort_by_key(|q| question_rank(&q.query));
    }
    let keys: Vec<String> = buckets.keys().cloned().collect();
    let mut cursors = vec![0usize; keys.len()];
    let mut out = Vec::new();
    while out.len() < max_pairs {
        let mut progressed = false;
        for (i, key) in keys.iter().enumerate() {
            let bucket = &buckets[key];
            if cursors[i] < bucket.len() {
                out.push(bucket[cursors[i]]);
                cursors[i] += 1;
                progressed = true;
                if out.len() >= max_pairs {
                    break;
                }
            }
        }
        if !progressed {
            break;
        }
    }
    out
}

fn subject_of(url: &str) -> String {
    let path = url.trim_start_matches('/');
    let path = path.strip_prefix("images/").unwrap_or(path);
    path.split(['/', '-', '.']).next().unwrap_or("").to_string()
}

fn question_rank(query: &str) -> u8 {
    let lower = query.to_lowercase();
    if lower.contains("how") {
        0
    } else if lower.starts_with("what") || lower.starts_with("explain") {
        1
    } else {
        2
    }
}

fn flatten_corpus(vectors: &[Vec<f32>]) -> Vec<f32> {
    let mut flat = Vec::with_capacity(vectors.len() * DIMS);
    for vector in vectors {
        flat.extend(vector.iter().take(DIMS).copied());
        if vector.len() < DIMS {
            flat.extend(std::iter::repeat(0f32).take(DIMS - vector.len()));
        }
    }
    flat
}

fn tile_to_batches(pairs: &[Pair], batch: usize) -> Vec<Pair> {
    if pairs.is_empty() || batch <= 1 {
        return pairs.to_vec();
    }
    let target = pairs.len().div_ceil(batch) * batch;
    let mut out = Vec::with_capacity(target);
    while out.len() < target {
        out.extend(pairs.iter().cloned());
    }
    out.truncate(target);
    out
}

/// How often to run the held-out forward. Burn validates once per learner epoch,
/// and a learner epoch used to be a single optimizer step.
const STEPS_BETWEEN_VALIDATIONS: usize = 10;

/// `(learner epochs, optimizer steps in each epoch)`.
/// Every epoch has the same number of steps, and that number divides the run,
/// so the cosine schedule still sees one step per optimizer update.
fn validation_cadence(optimizer_steps: usize, batches_per_pass: usize) -> (usize, usize) {
    let pass = batches_per_pass.max(1);
    let steps = optimizer_steps.max(pass);
    let mut per_epoch = STEPS_BETWEEN_VALIDATIONS.min(steps).max(pass);
    while per_epoch > pass && (steps % per_epoch != 0 || per_epoch % pass != 0) {
        per_epoch -= 1;
    }
    if steps % per_epoch != 0 || per_epoch % pass != 0 {
        per_epoch = pass;
    }
    (steps / per_epoch, per_epoch)
}

fn repeat_passes(pairs: &[Pair], times: usize) -> Vec<Pair> {
    let times = times.max(1);
    let mut out = Vec::with_capacity(pairs.len().saturating_mul(times));
    for _ in 0..times {
        out.extend(pairs.iter().cloned());
    }
    out
}

/// Linear warmup over the first tenth of the run, then cosine decay from the peak to 1% of it.
fn cosine_schedule(peak: f64, steps: usize) -> Result<ModuleLrScheduler, String> {
    let steps = steps.max(1);
    let floor = (peak * 0.01).clamp(1e-6, peak);
    if steps == 1 {
        return CosineAnnealingLrSchedulerConfig::new(peak, 1).with_min_lr(floor).init();
    }
    let warmup = (steps / 10).clamp(1, steps - 1);
    let decay = (steps - warmup).max(1);
    SequentialLrSchedulerConfig::new(
        vec![
            LrSchedulerConfig::Linear(LinearLrSchedulerConfig::new(floor, peak, warmup)),
            LrSchedulerConfig::Cosine(CosineAnnealingLrSchedulerConfig::new(peak, decay).with_min_lr(floor)),
        ],
        vec![warmup],
    )
    .init()
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

pub fn distill(
    chunks: &[Chunk],
    vectors: &[Vec<f32>],
    queries: &[QueryRecord],
    tokenizer: &CompactBpe,
    embed_query: &dyn Fn(&str) -> Vec<f32>,
    config: &DistillConfig,
) -> Result<StudentWeights, String> {
    let mut pairs = build_pairs(chunks, vectors, queries, tokenizer, embed_query, config.max_pairs);
    if pairs.len() < 4 {
        return Err(format!("need at least 4 training pairs, found {}", pairs.len()));
    }
    let valid_n = (pairs.len() / 5).max(1);
    let valid = pairs.split_off(pairs.len() - valid_n);
    let batch_size = config.batch_size.max(1);
    let contrast_size = batch_size.max(MIN_CONTRAST);
    let train_pairs = tile_to_batches(&pairs, batch_size);
    let train_batch = batch_size.min(train_pairs.len()).max(1);
    let batches_per_pass = train_pairs.len().div_ceil(train_batch);
    let total_steps = batches_per_pass * config.epochs.max(1);
    // Several optimizer steps share one learner epoch. The schedule still
    // advances on every step; validation runs once per epoch instead of once per step.
    let (num_epochs, steps_per_epoch) = validation_cadence(total_steps, batches_per_pass);
    let epoch_pairs = repeat_passes(&train_pairs, steps_per_epoch / batches_per_pass);
    let device = Device::default();
    let strategy = ExecutionStrategy::single(device.clone());
    let train_device = strategy.main_device().clone().autodiff();
    let model = QueryEncoderConfig {
        vocab: config.vocab,
        proj: 16,
        hidden: 128,
        layers: 2,
        heads: 4,
        intermediate: 512,
        max_len: 128,
        out_dim: DIMS,
    }
    .init(&train_device);
    let flat = flatten_corpus(vectors);
    let batcher = PairBatcher {
        max_len: 128,
        pad_id: tokenizer.pad_id as i32,
        n_corpus: vectors.len(),
        corpus: Arc::new(flat),
        contrast_size,
        rng: Arc::new(Mutex::new(Lcg(0xC0FFEE))),
    };
    let train_loader = DataLoaderBuilder::new(batcher.clone())
        .batch_size(train_batch)
        .shuffle(0x5EED)
        .num_workers(0)
        .build(InMemDataset::new(epoch_pairs));
    // One batch for the whole held-out set, so the forward is a single device launch.
    let valid_loader = DataLoaderBuilder::new(batcher)
        .batch_size(valid.len().max(1))
        .num_workers(0)
        .build(InMemDataset::new(valid));
    let optim = AdamWConfig::new()
        .with_weight_decay(1e-4)
        .with_beta_2(0.98)
        .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
        .init();
    let lr = cosine_schedule(config.lr, total_steps)?;
    std::fs::create_dir_all(&config.artifact_dir).map_err(|e| e.to_string())?;
    let training = SupervisedTraining::new(
        config.artifact_dir.to_string_lossy().as_ref(),
        train_loader,
        valid_loader,
    )
    .metric_train_numeric(burn::train::metric::LossMetric::new())
    .metric_valid_numeric(burn::train::metric::LossMetric::new())
    .metric_train_numeric(ScalarMetric::<CosineIn>::new("cosine"))
    .metric_valid_numeric(ScalarMetric::<RecallIn>::new("recall@1"))
    .metric_train_numeric(burn::train::metric::LearningRateMetric::new())
    .with_training_strategy(strategy.into())
    .num_epochs(num_epochs)
    .summary();
    let trained = training.launch(Learner::new(model, optim, lr));
    Ok(trained.model.export_weights())
}

#[cfg(test)]
mod tests {
    use super::*;
    use remixsearch_build::Chunk;
    use remixsearch_format::{ImageRef, QueryRecord};

    #[test]
    fn one_epoch_leaves_a_finite_loss() {
        let chunks: Vec<Chunk> = (0..6)
            .map(|i| Chunk {
                url: format!("/doc/{i}"),
                title: format!("signal {i}"),
                headings: vec![format!("heading {i}")],
                text: format!("signal {i} uses scale 0.{i} and a private window"),
                images: vec![ImageRef { src: format!("images/{i}.png"), alt: format!("diagram of signal {i}"), caption: String::new() }],
            })
            .collect();
        let vectors: Vec<Vec<f32>> = chunks.iter().map(crate::embed_chunk_hash).collect();
        let queries: Vec<QueryRecord> = chunks
            .iter()
            .map(|c| QueryRecord {
                query: format!("what is {}?", c.title),
                positive_url: c.url.clone(),
                kind: "passage".into(),
                held_out: false,
                unanswerable: false,
            })
            .collect();
        let tok = crate::train_tokenizer(&chunks.iter().map(|c| c.text.clone()).collect::<Vec<_>>(), 128).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let config = DistillConfig { epochs: 1, batch_size: 4, max_pairs: 8, lr: 1e-3, vocab: 128, artifact_dir: dir.path().to_path_buf() };
        let teacher = HashTeacher;
        let weights = distill(&chunks, &vectors, &queries, &tok, &|q| teacher.embed(q), &config).unwrap();
        assert_eq!(weights.out_dim, DIMS);
        assert!(weights.out_w.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn how_questions_survive_a_small_pair_budget() {
        let dir = tempfile::tempdir().unwrap();
        let paths = crate::generate_corpus(dir.path(), 2, false).unwrap();
        let queries = remixsearch_build::read_queries(&paths.queries).unwrap();
        let selected = select_queries(&queries, 32);
        assert!(selected.iter().any(|q| q.query == "how does the bm25 working"));
        let positives: Vec<_> = queries
            .iter()
            .filter(|q| q.query == "how does the bm25 working")
            .map(|q| q.positive_url.as_str())
            .collect();
        assert_eq!(positives, vec!["/bm25/how-it-works#a"]);
    }

    #[test]
    fn contrastive_draws_change_and_start_at_2048() {
        let mut corpus = Vec::new();
        for i in 0..4 {
            let mut vector = vec![0f32; DIMS];
            vector[i] = 1.0;
            corpus.extend(vector);
        }
        let batcher = PairBatcher {
            max_len: 8,
            pad_id: 0,
            corpus: Arc::new(corpus),
            n_corpus: 4,
            contrast_size: MIN_CONTRAST,
            rng: Arc::new(Mutex::new(Lcg(1))),
        };
        let first = batcher.sample_negatives();
        let second = batcher.sample_negatives();
        assert_eq!(first.len(), MIN_CONTRAST * DIMS);
        assert_ne!(first, second);
        let tiled = tile_to_batches(
            &(0..3)
                .map(|i| Pair { tokens: vec![i], teacher: vec![0.0; DIMS], positive: vec![0.0; DIMS] })
                .collect::<Vec<_>>(),
            MIN_CONTRAST,
        );
        assert_eq!(tiled.len(), MIN_CONTRAST);
        let (epochs, steps) = validation_cadence(80, 1);
        assert_eq!((epochs, steps), (8, 10));
        assert_eq!(epochs * steps, 80);
        let (once, only) = validation_cadence(1, 1);
        assert_eq!((once, only), (1, 1));
    }

    #[test]
    fn cosine_schedule_warms_up_then_anneals() {
        let mut schedule = cosine_schedule(1e-3, 80).unwrap();
        let mut rates = Vec::new();
        for _ in 0..80 {
            rates.push(schedule.step().base());
        }
        let peak_at = rates.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
        assert!(peak_at > 0 && peak_at < 40, "peak landed at step {peak_at}");
        assert!(rates[peak_at] > 9e-4);
        assert!(rates[79] < rates[peak_at] * 0.2, "final lr {} did not anneal", rates[79]);
        assert!(rates[0] < rates[peak_at]);
    }
}

pub struct HashTeacher;

impl HashTeacher {
    pub fn embed(&self, text: &str) -> Vec<f32> {
        let mut v = crate::hash_embed(text);
        l2_normalize(&mut v);
        v
    }
}
