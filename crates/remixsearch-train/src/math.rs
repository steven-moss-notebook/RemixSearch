/// Scalar RMSNorm, RoPE, and Gemma 4 image patching. The Burn teacher uses the
/// same formulas; these functions lock them without a device.

pub fn rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let mean_sq = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = (mean_sq + eps).powf(-0.5);
    x.iter()
        .zip(weight)
        .map(|(v, w)| v * inv * w)
        .collect()
}

/// Default RoPE cos/sin for one position. `head_dim` is even.
pub fn rope_cos_sin(position: f32, head_dim: usize, theta: f32) -> (Vec<f32>, Vec<f32>) {
    let mut freq = Vec::with_capacity(head_dim / 2);
    for i in (0..head_dim).step_by(2) {
        let exp = i as f32 / head_dim as f32;
        freq.push(1.0 / theta.powf(exp));
    }
    let mut cos = Vec::with_capacity(head_dim);
    let mut sin = Vec::with_capacity(head_dim);
    for f in &freq {
        let angle = position * f;
        cos.push(angle.cos());
        sin.push(angle.sin());
    }
    let mut cos_full = cos.clone();
    cos_full.extend_from_slice(&cos);
    let mut sin_full = sin.clone();
    sin_full.extend_from_slice(&sin);
    (cos_full, sin_full)
}

pub fn rotate_half_apply(x: &[f32], cos: &[f32], sin: &[f32]) -> Vec<f32> {
    let n = x.len();
    let half = n / 2;
    let mut rot = vec![0f32; n];
    for i in 0..half {
        rot[i] = -x[half + i];
        rot[half + i] = x[i];
    }
    x.iter()
        .zip(cos)
        .zip(rot.iter().zip(sin))
        .map(|((a, c), (r, s))| a * c + r * s)
        .collect()
}

/// Largest aspect-preserving size whose sides are multiples of `pooling * patch`
/// and whose patch count does not exceed `max_patches`.
pub fn aspect_size(height: u32, width: u32, patch: u32, max_patches: u32, pooling: u32) -> (u32, u32) {
    let side = pooling * patch;
    let target_px = max_patches as f32 * (patch * patch) as f32;
    let total = (height * width).max(1) as f32;
    let factor = (target_px / total).sqrt();
    let mut th = ((factor * height as f32) / side as f32).floor() as u32 * side;
    let mut tw = ((factor * width as f32) / side as f32).floor() as u32 * side;
    if th == 0 {
        th = side;
    }
    if tw == 0 {
        tw = side;
    }
    while th * tw > max_patches * patch * patch && (th > side || tw > side) {
        if th >= tw && th > side {
            th -= side;
        } else if tw > side {
            tw -= side;
        } else {
            break;
        }
    }
    (th, tw)
}

/// RGB f32 image in [0, 1], row-major, into patches of `patch * patch * 3`
/// and (x, y) patch coordinates. Patches are stored in row-major (y, x) order.
pub fn patchify(rgb: &[f32], width: u32, height: u32, patch: u32) -> (Vec<f32>, Vec<(i32, i32)>) {
    let pw = width / patch;
    let ph = height / patch;
    let mut values = Vec::new();
    let mut pos = Vec::new();
    for y in 0..ph {
        for x in 0..pw {
            for py in 0..patch {
                for px in 0..patch {
                    let xx = x * patch + px;
                    let yy = y * patch + py;
                    let base = ((yy * width + xx) * 3) as usize;
                    values.extend_from_slice(&rgb[base..base + 3]);
                }
            }
            pos.push((x as i32, y as i32));
        }
    }
    (values, pos)
}

/// 3×3 spatial average of patch vectors, matching Gemma 4's pool when the grid
/// divides evenly. `patch_dim` is the width of each patch vector.
pub fn pool3(patches: &[f32], grid_w: usize, grid_h: usize, dim: usize) -> Vec<f32> {
    let out_w = grid_w / 3;
    let out_h = grid_h / 3;
    let mut out = vec![0f32; out_w * out_h * dim];
    for oy in 0..out_h {
        for ox in 0..out_w {
            for ky in 0..3 {
                for kx in 0..3 {
                    let sy = oy * 3 + ky;
                    let sx = ox * 3 + kx;
                    let src = (sy * grid_w + sx) * dim;
                    let dst = (oy * out_w + ox) * dim;
                    for d in 0..dim {
                        out[dst + d] += patches[src + d] / 9.0;
                    }
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rms_matches_definition() {
        let y = rms_norm(&[3.0, 4.0], &[1.0, 1.0], 1e-6);
        let inv = (12.5 + 1e-6_f32).powf(-0.5);
        assert!((y[0] - 3.0 * inv).abs() < 1e-5);
        assert!((y[1] - 4.0 * inv).abs() < 1e-5);
    }

    #[test]
    fn rope_identity_at_origin() {
        let (cos, sin) = rope_cos_sin(0.0, 4, 10_000.0);
        assert!(cos.iter().all(|c| (c - 1.0).abs() < 1e-5));
        assert!(sin.iter().all(|s| s.abs() < 1e-5));
        let x = [0.2, -0.4, 0.6, 0.1];
        let y = rotate_half_apply(&x, &cos, &sin);
        for (a, b) in x.iter().zip(&y) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn resize_and_pool_lock() {
        let (h, w) = aspect_size(96, 48, 16, 2520, 3);
        assert_eq!(h % 48, 0);
        assert_eq!(w % 48, 0);
        assert!(h * w <= 2520 * 256);
        let rgb = vec![0.5f32; 48 * 48 * 3];
        let (patches, pos) = patchify(&rgb, 48, 48, 16);
        assert_eq!(pos.len(), 9);
        assert_eq!(pos[0], (0, 0));
        assert_eq!(pos[3], (0, 1));
        let pooled = pool3(&patches, 3, 3, 16 * 16 * 3);
        assert_eq!(pooled.len(), 16 * 16 * 3);
        assert!((pooled[0] - 0.5).abs() < 1e-4);
    }
}
