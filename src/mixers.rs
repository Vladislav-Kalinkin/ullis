use anyhow::Result;
use candle_core::{DType, Device, Tensor, Var, D};

/// Parameter-free token mix: delay half the channels by one step.
pub fn causal_shift(x: &Tensor) -> Result<Tensor> {
    let (b, t, c) = x.dims3()?;
    let split = c / 2;
    let keep = x.narrow(D::Minus1, 0, split)?;
    let delayed = x.narrow(D::Minus1, split, c - split)?;
    let delayed = if t <= 1 {
        Tensor::zeros_like(&delayed)?
    } else {
        let body = delayed.narrow(1, 0, t - 1)?;
        let zeros = Tensor::zeros((b, 1, c - split), x.dtype(), x.device())?;
        Tensor::cat(&[&zeros, &body], 1)?
    };
    Ok(Tensor::cat(&[&keep, &delayed], D::Minus1)?)
}

pub struct CausalAttention {
    pub n_heads: usize,
    pub head_dim: usize,
    pub qkv: Var,  // [3d, d]
    pub proj: Var, // [d, d]
}

impl CausalAttention {
    pub fn new(
        d_model: usize,
        n_heads: usize,
        device: &Device,
        rng: &mut impl rand::Rng,
    ) -> Result<Self> {
        if d_model % n_heads != 0 {
            anyhow::bail!("d_model must be divisible by n_heads");
        }
        let head_dim = d_model / n_heads;
        let qkv = rand_kaiming(3 * d_model, d_model, device, rng)?;
        let proj = rand_kaiming(d_model, d_model, device, rng)?;
        Ok(Self {
            n_heads,
            head_dim,
            qkv: Var::from_tensor(&qkv)?,
            proj: Var::from_tensor(&proj)?,
        })
    }

    pub fn vars(&self) -> Vec<Var> {
        vec![self.qkv.clone(), self.proj.clone()]
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, t, d) = x.dims3()?;
        let qkv = x.matmul(&self.qkv.as_tensor().t()?)?; // [B,T,3d]
        let qkv = qkv.reshape((b, t, 3, self.n_heads, self.head_dim))?;
        let q = qkv.narrow(2, 0, 1)?.squeeze(2)?.transpose(1, 2)?; // [B,H,T,hd]
        let k = qkv.narrow(2, 1, 1)?.squeeze(2)?.transpose(1, 2)?;
        let v = qkv.narrow(2, 2, 1)?.squeeze(2)?.transpose(1, 2)?;
        let scale = (self.head_dim as f64).sqrt().recip();
        let attn = q.matmul(&k.transpose(D::Minus1, D::Minus2)?)?;
        let attn = (attn * scale)?;
        let causal = causal_mask(t, x.dtype(), x.device())?;
        let attn = attn.broadcast_add(&causal)?;
        let attn = candle_nn::ops::softmax(&attn, D::Minus1)?;
        let out = attn.matmul(&v)?.transpose(1, 2)?.reshape((b, t, d))?;
        Ok(out.matmul(&self.proj.as_tensor().t()?)?)
    }
}

fn causal_mask(t: usize, dtype: DType, device: &Device) -> Result<Tensor> {
    let mut v = vec![0f32; t * t];
    for i in 0..t {
        for j in 0..t {
            if j > i {
                v[i * t + j] = f32::NEG_INFINITY;
            }
        }
    }
    Ok(Tensor::from_vec(v, (t, t), device)?.to_dtype(dtype)?)
}

pub fn rand_kaiming(
    out: usize,
    in_f: usize,
    device: &Device,
    rng: &mut impl rand::Rng,
) -> Result<Tensor> {
    let bound = (in_f as f32).sqrt().recip();
    let n = out * in_f;
    let data: Vec<f32> = (0..n).map(|_| rng.random_range(-bound..bound)).collect();
    Ok(Tensor::from_vec(data, (out, in_f), device)?)
}

pub fn rand_uniform(
    shape: &[usize],
    lo: f32,
    hi: f32,
    device: &Device,
    rng: &mut impl rand::Rng,
) -> Result<Tensor> {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n).map(|_| rng.random_range(lo..hi)).collect();
    Ok(Tensor::from_vec(data, shape, device)?)
}

pub fn randn(
    shape: &[usize],
    std: f32,
    device: &Device,
    rng: &mut impl rand::Rng,
) -> Result<Tensor> {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n)
        .map(|_| {
            let u: f32 = rng.random::<f32>().max(1e-7);
            let v: f32 = rng.random::<f32>();
            std * (-2.0 * u.ln()).sqrt() * (2.0 * std::f32::consts::PI * v).cos()
        })
        .collect();
    Ok(Tensor::from_vec(data, shape, device)?)
}
