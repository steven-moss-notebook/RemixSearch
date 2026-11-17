use std::collections::HashSet;

pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut d = 0.0f32;
    for i in 0..n {
        d += a[i] * b[i];
    }
    d
}

pub fn l2_normalize(v: &mut [f32]) {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 1e-8 {
        for x in v {
            *x /= n;
        }
    }
}

/// Greedy HNSW layer search. Returns ids sorted by descending cosine.
pub fn search_layer<N, V>(query: &[f32], entry: u32, ef: usize, neighbors: &N, vector: &V) -> Vec<(u32, f32)>
where
    N: Fn(u32) -> Vec<u32>,
    V: Fn(u32) -> Vec<f32>,
{
    let ef = ef.max(1);
    let mut best = vec![(entry, cosine(query, &vector(entry)))];
    let mut cand = best.clone();
    let mut seen = HashSet::from([entry]);
    while !cand.is_empty() {
        cand.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let (id, sim) = cand.remove(0);
        let worst = best.last().map(|(_, s)| *s).unwrap_or(f32::MIN);
        if best.len() >= ef && sim < worst {
            break;
        }
        for nb in neighbors(id) {
            if !seen.insert(nb) {
                continue;
            }
            let s = cosine(query, &vector(nb));
            if best.len() < ef || s > best.last().map(|(_, x)| *x).unwrap_or(f32::MIN) {
                cand.push((nb, s));
                best.push((nb, s));
                best.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                if best.len() > ef {
                    best.truncate(ef);
                }
            }
        }
    }
    best
}

#[derive(Clone, Debug)]
pub struct HnswGraph {
    pub level0: Vec<Vec<u32>>,
    pub upper: Vec<Vec<Vec<u32>>>,
    pub entry: u32,
}

impl HnswGraph {
    pub fn build(vecs: &[Vec<f32>], m: usize, ef_construction: usize, seed: u64) -> Self {
        let n = vecs.len();
        if n == 0 {
            return Self { level0: Vec::new(), upper: Vec::new(), entry: 0 };
        }
        let ml = 1.0 / (m.max(2) as f64).ln();
        let mut rng = seed | 1;
        let mut levels = Vec::with_capacity(n);
        for _ in 0..n {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let unit = ((rng >> 11) as f64) / ((1u64 << 53) as f64);
            let level = ((-unit.max(1e-12).ln()) * ml).floor() as usize;
            levels.push(level.min(6));
        }
        let mut level0 = vec![Vec::new(); n];
        let mut max_level = 0usize;
        let mut entry = 0u32;
        let mut upper: Vec<Vec<Vec<u32>>> = Vec::new();
        let vector = |id: u32| vecs[id as usize].clone();
        for i in 0..n {
            let id = i as u32;
            let level = levels[i];
            while upper.len() < level {
                upper.push(vec![Vec::new(); n]);
            }
            if i == 0 {
                max_level = level;
                entry = id;
                continue;
            }
            let mut ep = entry;
            if level < max_level {
                for lv in (level + 1..=max_level).rev() {
                    let neigh = |nid: u32| {
                        if lv == 0 {
                            level0[nid as usize].clone()
                        } else {
                            upper[lv - 1][nid as usize].clone()
                        }
                    };
                    let found = search_layer(&vecs[i], ep, 1, &neigh, &vector);
                    if let Some((nid, _)) = found.first() {
                        ep = *nid;
                    }
                }
            }
            for lv in (0..=level.min(max_level)).rev() {
                let limit = if lv == 0 { m * 2 } else { m };
                let neigh = |nid: u32| {
                    if lv == 0 {
                        level0[nid as usize].clone()
                    } else {
                        upper[lv - 1][nid as usize].clone()
                    }
                };
                let found = search_layer(&vecs[i], ep, ef_construction, &neigh, &vector);
                let selected: Vec<u32> = found.into_iter().take(limit).map(|(nid, _)| nid).collect();
                if lv == 0 {
                    connect(&mut level0, id, &selected, limit, vecs);
                } else {
                    connect(&mut upper[lv - 1], id, &selected, limit, vecs);
                }
                if let Some(nid) = selected.first() {
                    ep = *nid;
                }
            }
            if level > max_level {
                max_level = level;
                entry = id;
            }
        }
        Self { level0, upper, entry }
    }
}

fn connect(layer: &mut [Vec<u32>], id: u32, selected: &[u32], limit: usize, vecs: &[Vec<f32>]) {
    for &nb in selected {
        if nb == id {
            continue;
        }
        layer[id as usize].push(nb);
        layer[nb as usize].push(id);
        prune(&mut layer[nb as usize], nb, limit, vecs);
    }
    prune(&mut layer[id as usize], id, limit, vecs);
}

fn prune(neigh: &mut Vec<u32>, id: u32, limit: usize, vecs: &[Vec<f32>]) {
    neigh.sort_unstable();
    neigh.dedup();
    neigh.retain(|n| *n != id);
    if neigh.len() > limit {
        neigh.sort_by(|a, b| {
            let da = cosine(&vecs[id as usize], &vecs[*a as usize]);
            let db = cosine(&vecs[id as usize], &vecs[*b as usize]);
            db.partial_cmp(&da).unwrap_or(std::cmp::Ordering::Equal)
        });
        neigh.truncate(limit);
    }
}
