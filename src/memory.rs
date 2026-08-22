//! Ullis Memory model: FWHT + gated scan + top-k ternary experts + slots.

use anyhow::{bail, Result};

use crate::config::TrainConfig;
use crate::device::SovereignDevice;
use crate::expert::{
    l1_experts, l1_experts_grad, moe_backward, moe_forward, MoeCache, TernaryExpert,
};
use crate::hadamard::{fwht_rows, fwht_rows_bwd};
use crate::mixers::{
    embed_lookup_into, embed_scatter_acc, randn, rmsnorm_backward_into, rmsnorm_into,
    streamed_tied_ce_acc,
};
use crate::optim::DenseSgd;
use crate::scan::{scan_backward, scan_forward, scan_step, ScanParams, ScanTape};
use crate::slots::{slots_backward, slots_forward, slots_step, SlotParams, SlotState, SlotTape};

#[derive(Clone, Debug)]
pub struct HostNorm {
    pub weight: Vec<f32>,
    pub grad: Vec<f32>,
    pub eps: f32,
}

impl HostNorm {
    fn new(d: usize) -> Self {
        Self {
            weight: vec![1.0; d],
            grad: vec![0.0; d],
            eps: 1e-5,
        }
    }

    fn zero_grad(&mut self) {
        self.grad.fill(0.0);
    }
}

#[derive(Debug)]
pub struct MemoryBlock {
    pub n1: HostNorm,
    pub n2: HostNorm,
    pub n_slot: HostNorm,
    pub scan: ScanParams,
    pub experts: Vec<TernaryExpert>,
    pub router: Vec<f32>,
    pub grad_router: Vec<f32>,
    pub slots: Option<SlotParams>,
}

impl MemoryBlock {
    fn new(cfg: &TrainConfig, layer: usize, rng: &mut impl rand::Rng) -> Self {
        let d = cfg.d_model;
        let e = cfg.mem_experts;
        let w = cfg.expert_width.max(1);
        let experts = (0..e).map(|_| TernaryExpert::new(d, w, rng)).collect();
        let router = if e == 0 {
            Vec::new()
        } else {
            crate::mixers::rand_kaiming(e, d, rng)
        };
        let slots = if layer % 2 == 0 && cfg.n_slots > 0 {
            let mut sp = SlotParams::new(d, cfg.n_slots);
            // Residual injection starts small so scan/experts are not drowned
            // while content addressing is still uniform (empty keys).
            sp.gamma = 0.1;
            Some(sp)
        } else {
            None
        };
        Self {
            n1: HostNorm::new(d),
            n2: HostNorm::new(d),
            n_slot: HostNorm::new(d),
            scan: ScanParams::new(d),
            experts,
            router,
            grad_router: vec![0.0; e * d],
            slots,
        }
    }

    fn set_phase(&mut self, phase: u8) {
        for e in &mut self.experts {
            e.phase = phase;
        }
    }

    fn zero_grad(&mut self) {
        self.n1.zero_grad();
        self.n2.zero_grad();
        self.n_slot.zero_grad();
        self.scan.zero_grad();
        self.grad_router.fill(0.0);
        for e in &mut self.experts {
            e.zero_grad();
        }
        if let Some(s) = self.slots.as_mut() {
            s.zero_grad();
        }
    }
}

#[derive(Debug)]
struct BlockTape {
    n1_in: Vec<f32>,
    scan: ScanTape,
    after_scan: Vec<f32>,
    moe: MoeCache,
    after_ff: Vec<f32>,
    slot: Option<SlotTape>,
}

#[derive(Debug)]
pub struct LayerCache {
    pub h: Vec<f32>,
    pub prev_u: Vec<f32>,
    pub slots: Option<SlotState>,
}

#[derive(Debug)]
pub struct UllisMemory {
    pub cfg: TrainConfig,
    pub embed: Vec<f32>,
    pub embed_grad: Vec<f32>,
    pub norm: HostNorm,
    pub blocks: Vec<MemoryBlock>,
    pub phase: u8,
    pub last_ce: f32,
    pub last_entropy: f32,
    pub last_mask: f32,
    pub last_fwd_ms: f32,
    pub last_ce_ms: f32,
    pub last_bwd_ms: f32,
    pub device: SovereignDevice,
    tapes: Vec<BlockTape>,
}

impl UllisMemory {
    pub fn new(cfg: TrainConfig, rng: &mut impl rand::Rng) -> Result<Self> {
        Self::with_device(cfg, rng, SovereignDevice::open(false)?)
    }

    pub fn with_device(
        cfg: TrainConfig,
        rng: &mut impl rand::Rng,
        device: SovereignDevice,
    ) -> Result<Self> {
        if cfg.d_model == 0 {
            bail!("d_model == 0");
        }
        let d = cfg.d_model;
        let v = cfg.vocab_size;
        let embed = randn(v * d, 0.02, rng);
        let n_layers = cfg.n_layers.max(1);
        let blocks = (0..n_layers)
            .map(|i| MemoryBlock::new(&cfg, i, rng))
            .collect();
        Ok(Self {
            cfg,
            embed,
            embed_grad: vec![0.0; v * d],
            norm: HostNorm::new(d),
            blocks,
            phase: 1,
            last_ce: 0.0,
            last_entropy: 0.0,
            last_mask: 0.0,
            last_fwd_ms: 0.0,
            last_ce_ms: 0.0,
            last_bwd_ms: 0.0,
            device,
            tapes: Vec::new(),
        })
    }

    pub fn set_phase(&mut self, phase: u8) {
        self.phase = phase;
        for b in &mut self.blocks {
            b.set_phase(phase);
        }
    }

    pub fn param_report(&self) -> String {
        let mut n = self.embed.len() + self.norm.weight.len();
        for b in &self.blocks {
            n += b.n1.weight.len() + b.n2.weight.len() + b.n_slot.weight.len();
            n += b.scan.w_alpha.len() * 4 + b.router.len();
            for e in &b.experts {
                n += e.w_up.len() + e.w_gate.len() + e.w_down.len() + e.bumps.len();
            }
            if let Some(s) = &b.slots {
                n += s.w_q.len() + s.w_w.len() + s.w_link.len() + 8;
            }
        }
        format!(
            "memory D={} L={} E={} W={} S={} params={}",
            self.cfg.d_model,
            self.blocks.len(),
            self.cfg.mem_experts,
            self.cfg.expert_width,
            self.cfg.n_slots,
            n
        )
    }

    fn zero_grad(&mut self) {
        self.embed_grad.fill(0.0);
        self.norm.zero_grad();
        for b in &mut self.blocks {
            b.zero_grad();
        }
    }

    fn block_forward(
        &mut self,
        li: usize,
        x: &[f32],
        b: usize,
        t: usize,
        tape: bool,
    ) -> Result<Vec<f32>> {
        let d = self.cfg.d_model;
        let n = b * t;
        let blk = &self.blocks[li];
        let mut n1 = vec![0.0f32; n * d];
        rmsnorm_into(x, n, d, &blk.n1.weight, blk.n1.eps, &mut n1)?;
        let mut had = vec![0.0f32; n * d];
        fwht_rows(&n1, n, d, &mut had)?;
        let (h_scan, scan_tape, _) = scan_forward(&blk.scan, &had, b, t, None)?;
        let mut after = vec![0.0f32; n * d];
        for i in 0..n * d {
            after[i] = x[i] + h_scan[i];
        }
        let mut n2 = vec![0.0f32; n * d];
        rmsnorm_into(&after, n, d, &blk.n2.weight, blk.n2.eps, &mut n2)?;
        let k = self.cfg.moe_topk.max(1) as usize;
        let gpu = if self.device.is_metal() {
            Some(&self.device)
        } else {
            None
        };
        let (ff, moe) = moe_forward(&blk.experts, &blk.router, &n2, n, d, k, gpu)?;
        let mut after_ff = vec![0.0f32; n * d];
        for i in 0..n * d {
            after_ff[i] = after[i] + ff[i];
        }
        let (y, slot_tape, after_ff_tape) = if let Some(sp) = blk.slots.as_ref() {
            let mut slot_n = vec![0.0f32; n * d];
            rmsnorm_into(
                &after_ff,
                n,
                d,
                &blk.n_slot.weight,
                blk.n_slot.eps,
                &mut slot_n,
            )?;
            let mut st = SlotState::new(b, self.cfg.n_slots, d);
            let (sv, tp) = slots_forward(sp, &slot_n, b, t, &mut st)?;
            let mut out = after_ff.clone();
            for i in 0..n * d {
                out[i] += sv[i];
            }
            (out, Some(tp), after_ff)
        } else {
            (after_ff, None, Vec::new())
        };
        if tape {
            self.tapes.push(BlockTape {
                n1_in: x.to_vec(),
                scan: scan_tape,
                after_scan: after,
                moe,
                after_ff: after_ff_tape,
                slot: slot_tape,
            });
        }
        Ok(y)
    }

    fn block_backward(&mut self, li: usize, dy: &[f32], b: usize, t: usize) -> Result<Vec<f32>> {
        let d = self.cfg.d_model;
        let n = b * t;
        let aux = self.cfg.moe_aux as f32;
        let tape = self.tapes.pop().expect("block tape");
        let blk = &mut self.blocks[li];
        let mut dy_ff = dy.to_vec();
        if let Some(slots) = blk.slots.as_mut() {
            let st = tape.slot.as_ref().expect("slot tape");
            let dslot = slots_backward(slots, st, dy, b, t)?;
            let mut dn = vec![0.0f32; n * d];
            let mut dw = vec![0.0f32; d];
            rmsnorm_backward_into(
                &tape.after_ff,
                &dslot,
                n,
                d,
                &blk.n_slot.weight,
                blk.n_slot.eps,
                &mut dn,
                &mut dw,
            )?;
            for c in 0..d {
                blk.n_slot.grad[c] += dw[c];
            }
            for i in 0..n * d {
                dy_ff[i] += dn[i];
            }
        }
        let dn2_from_moe = moe_backward(
            &mut blk.experts,
            &blk.router,
            &mut blk.grad_router,
            &tape.moe,
            &dy_ff,
            n,
            d,
            aux,
        )?;
        let mut dn2 = vec![0.0f32; n * d];
        let mut dw = vec![0.0f32; d];
        rmsnorm_backward_into(
            &tape.after_scan,
            &dn2_from_moe,
            n,
            d,
            &blk.n2.weight,
            blk.n2.eps,
            &mut dn2,
            &mut dw,
        )?;
        for c in 0..d {
            blk.n2.grad[c] += dw[c];
        }
        let mut d_after = vec![0.0f32; n * d];
        for i in 0..n * d {
            d_after[i] = dy_ff[i] + dn2[i];
        }
        let dhad = scan_backward(&mut blk.scan, &tape.scan, &d_after, b, t)?;
        let mut dn1 = vec![0.0f32; n * d];
        fwht_rows_bwd(&dhad, n, d, &mut dn1)?;
        let mut dx = vec![0.0f32; n * d];
        dw.fill(0.0);
        rmsnorm_backward_into(
            &tape.n1_in,
            &dn1,
            n,
            d,
            &blk.n1.weight,
            blk.n1.eps,
            &mut dx,
            &mut dw,
        )?;
        for c in 0..d {
            blk.n1.grad[c] += dw[c];
        }
        for i in 0..n * d {
            dx[i] += d_after[i];
        }
        Ok(dx)
    }

    pub fn train_step(
        &mut self,
        ids: &[u32],
        targets: &[u32],
        mask: &[u8],
        b: usize,
        t: usize,
        l1: f32,
    ) -> Result<f32> {
        self.zero_grad();
        self.tapes.clear();
        let d = self.cfg.d_model;
        let v = self.cfg.vocab_size;
        let n = b * t;
        let t0 = std::time::Instant::now();
        let mut x = vec![0.0f32; n * d];
        embed_lookup_into(&self.embed, v, d, ids, &mut x)?;
        for li in 0..self.blocks.len() {
            x = self.block_forward(li, &x, b, t, true)?;
        }
        let pre_norm = x;
        let mut hidden = vec![0.0f32; n * d];
        rmsnorm_into(
            &pre_norm,
            n,
            d,
            &self.norm.weight,
            self.norm.eps,
            &mut hidden,
        )?;
        self.last_fwd_ms = t0.elapsed().as_secs_f32() * 1e3;
        let mut dh = vec![0.0f32; n * d];
        let mut row = Vec::new();
        let t1 = std::time::Instant::now();
        let (mut loss, mean_h) = streamed_tied_ce_acc(
            &hidden,
            &self.embed,
            n,
            d,
            v,
            targets,
            mask,
            self.cfg.entropy_coef as f32,
            &mut dh,
            &mut self.embed_grad,
            &mut row,
        )?;
        self.last_ce_ms = t1.elapsed().as_secs_f32() * 1e3;
        self.last_ce = loss - (self.cfg.entropy_coef as f32).max(0.0) * mean_h;
        self.last_entropy = mean_h;
        let n_sup = mask.iter().filter(|&&m| m != 0).count();
        self.last_mask = n_sup as f32 / n.max(1) as f32;
        if l1 > 0.0 {
            let mut s = 0.0f32;
            for blk in &self.blocks {
                s += l1_experts(&blk.experts);
            }
            loss += l1 * s;
            for blk in &mut self.blocks {
                l1_experts_grad(&mut blk.experts, l1);
            }
        }
        let t2 = std::time::Instant::now();
        let mut dx = vec![0.0f32; n * d];
        let mut dw = vec![0.0f32; d];
        rmsnorm_backward_into(
            &pre_norm,
            &dh,
            n,
            d,
            &self.norm.weight,
            self.norm.eps,
            &mut dx,
            &mut dw,
        )?;
        for c in 0..d {
            self.norm.grad[c] += dw[c];
        }
        for li in (0..self.blocks.len()).rev() {
            dx = self.block_backward(li, &dx, b, t)?;
        }
        embed_scatter_acc(v, d, ids, &dx, &mut self.embed_grad)?;
        self.last_bwd_ms = t2.elapsed().as_secs_f32() * 1e3;
        Ok(loss)
    }

    pub fn grad_sq(&self) -> f32 {
        let mut sq = 0.0f32;
        let add = |sq: &mut f32, g: &[f32]| {
            for &v in g {
                *sq += v * v;
            }
        };
        add(&mut sq, &self.embed_grad);
        add(&mut sq, &self.norm.grad);
        for blk in &self.blocks {
            add(&mut sq, &blk.n1.grad);
            add(&mut sq, &blk.n2.grad);
            add(&mut sq, &blk.n_slot.grad);
            add(&mut sq, &blk.scan.grad_w_alpha);
            add(&mut sq, &blk.scan.grad_b_alpha);
            add(&mut sq, &blk.scan.grad_w_i);
            add(&mut sq, &blk.scan.grad_b_i);
            add(&mut sq, &blk.grad_router);
            for e in &blk.experts {
                if self.phase < 4 {
                    add(&mut sq, &e.grad_up);
                    add(&mut sq, &e.grad_gate);
                    add(&mut sq, &e.grad_down);
                    add(&mut sq, &e.grad_bumps);
                }
                add(&mut sq, &e.grad_scale_up);
                add(&mut sq, &e.grad_scale_gate);
                add(&mut sq, &e.grad_scale_down);
            }
            if let Some(s) = &blk.slots {
                add(&mut sq, &s.grad_w_q);
                add(&mut sq, &s.grad_w_w);
                add(&mut sq, &s.grad_w_link);
                sq += s.grad_b_link * s.grad_b_link;
                sq += s.grad_w_bk * s.grad_w_bk;
                sq += s.grad_b_bk * s.grad_b_bk;
                sq += s.grad_w_bv * s.grad_w_bv;
                sq += s.grad_b_bv * s.grad_b_bv;
                sq += s.grad_gamma * s.grad_gamma;
                sq += s.grad_b_alloc * s.grad_b_alloc;
            }
        }
        sq
    }

    pub fn param_lens(&self) -> Vec<usize> {
        let mut v = Vec::new();
        let mut i = 0usize;
        self.walk_lens(&mut v, &mut i);
        v
    }

    fn walk_lens(&self, v: &mut Vec<usize>, _i: &mut usize) {
        v.push(self.embed.len());
        v.push(self.norm.weight.len());
        for blk in &self.blocks {
            v.push(blk.n1.weight.len());
            v.push(blk.n2.weight.len());
            v.push(blk.n_slot.weight.len());
            v.push(blk.scan.w_alpha.len());
            v.push(blk.scan.b_alpha.len());
            v.push(blk.scan.w_i.len());
            v.push(blk.scan.b_i.len());
            if !blk.router.is_empty() {
                v.push(blk.router.len());
            }
            for e in &blk.experts {
                if self.phase < 4 {
                    v.push(e.w_up.len());
                    v.push(e.w_gate.len());
                    v.push(e.w_down.len());
                    v.push(e.bumps.len());
                }
                v.push(e.scale_up.len());
                v.push(e.scale_gate.len());
                v.push(e.scale_down.len());
            }
            if blk.slots.is_some() {
                v.extend_from_slice(&[
                    self.cfg.d_model,
                    self.cfg.d_model,
                    self.cfg.d_model,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                ]);
            }
        }
    }

    pub fn pack(&mut self) {
        for blk in &mut self.blocks {
            for e in &mut blk.experts {
                e.pack();
            }
        }
        self.phase = 4;
    }

    pub fn new_cache(&self) -> Vec<LayerCache> {
        let d = self.cfg.d_model;
        self.blocks
            .iter()
            .map(|blk| LayerCache {
                h: vec![0.0; d],
                prev_u: vec![0.0; d],
                slots: blk
                    .slots
                    .as_ref()
                    .map(|_| SlotState::new(1, self.cfg.n_slots, d)),
            })
            .collect()
    }

    pub fn feed_token(&self, id: u32, caches: &mut [LayerCache]) -> Result<Vec<f32>> {
        let d = self.cfg.d_model;
        let v = self.cfg.vocab_size;
        let mut x = vec![0.0f32; d];
        embed_lookup_into(&self.embed, v, d, &[id], &mut x)?;
        for (li, blk) in self.blocks.iter().enumerate() {
            let mut n1 = vec![0.0f32; d];
            rmsnorm_into(&x, 1, d, &blk.n1.weight, blk.n1.eps, &mut n1)?;
            let mut had = vec![0.0f32; d];
            fwht_rows(&n1, 1, d, &mut had)?;
            let (h, new_prev) = scan_step(&blk.scan, &had, &caches[li].prev_u, &caches[li].h, 1)?;
            caches[li].h.copy_from_slice(&h);
            caches[li].prev_u.copy_from_slice(&new_prev);
            for i in 0..d {
                x[i] += h[i];
            }
            let mut n2 = vec![0.0f32; d];
            rmsnorm_into(&x, 1, d, &blk.n2.weight, blk.n2.eps, &mut n2)?;
            let k = self.cfg.moe_topk.max(1) as usize;
            let gpu = if self.device.is_metal() {
                Some(&self.device)
            } else {
                None
            };
            let (ff, _) = moe_forward(&blk.experts, &blk.router, &n2, 1, d, k, gpu)?;
            for i in 0..d {
                x[i] += ff[i];
            }
            if let (Some(sp), Some(st)) = (blk.slots.as_ref(), caches[li].slots.as_mut()) {
                let mut ns = vec![0.0f32; d];
                rmsnorm_into(&x, 1, d, &blk.n_slot.weight, blk.n_slot.eps, &mut ns)?;
                let sv = slots_step(sp, &ns, 1, st)?;
                for i in 0..d {
                    x[i] += sv[i];
                }
            }
        }
        let mut h = vec![0.0f32; d];
        rmsnorm_into(&x, 1, d, &self.norm.weight, self.norm.eps, &mut h)?;
        Ok(h)
    }

    pub fn logits(&self, hidden: &[f32]) -> Result<Vec<f32>> {
        let d = self.cfg.d_model;
        let v = self.cfg.vocab_size;
        let mut z = vec![0.0f32; v];
        crate::accelerate::sgemm_nt(1, v, d, 1.0, hidden, &self.embed, 0.0, &mut z)?;
        Ok(z)
    }

    pub fn argmax(logits: &[f32]) -> u32 {
        let mut best = 0usize;
        let mut bv = f32::NEG_INFINITY;
        for (i, &z) in logits.iter().enumerate() {
            if z > bv {
                bv = z;
                best = i;
            }
        }
        best as u32
    }

    pub fn sample_token(logits: &[f32], temperature: f32, rng: &mut impl rand::Rng) -> u32 {
        if temperature <= 1e-5 {
            return Self::argmax(logits);
        }
        let inv = 1.0 / temperature;
        let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let w: Vec<f32> = logits.iter().map(|z| ((z - m) * inv).exp()).collect();
        let s: f32 = w.iter().sum();
        let mut u = rng.random::<f32>() * s.max(1e-20);
        for (i, &p) in w.iter().enumerate() {
            if u <= p {
                return i as u32;
            }
            u -= p;
        }
        (w.len().saturating_sub(1)) as u32
    }

    pub fn generate_last(&mut self, ids: &[u32]) -> Result<u32> {
        let mut caches = self.new_cache();
        let mut last_h = vec![0.0f32; self.cfg.d_model];
        for &id in ids {
            last_h = self.feed_token(id, &mut caches)?;
        }
        let z = self.logits(&last_h)?;
        Ok(Self::argmax(&z))
    }

    pub fn next_token(
        &mut self,
        ctx: &[u32],
        temperature: f32,
        rng: &mut impl rand::Rng,
        caches: &mut Vec<LayerCache>,
        fed: &mut usize,
    ) -> Result<u32> {
        if *fed > ctx.len() {
            *caches = self.new_cache();
            *fed = 0;
        }
        while *fed < ctx.len() {
            let h = self.feed_token(ctx[*fed], caches)?;
            *fed += 1;
            if *fed == ctx.len() {
                let mut z = self.logits(&h)?;
                ban_recent(&mut z, ctx);
                return Ok(Self::sample_token(&z, temperature, rng));
            }
        }
        bail!("next_token empty context")
    }

    pub fn collect_blobs(&self) -> Vec<(String, crate::kan::NamedBlob)> {
        use crate::kan::NamedBlob;
        use crate::quant::{codes_to_i8, pack_ternary};
        let mut out = Vec::new();
        let push_f32 =
            |out: &mut Vec<(String, NamedBlob)>, name: String, data: &[f32], shape: Vec<usize>| {
                out.push((
                    name,
                    NamedBlob::F32 {
                        data: data.to_vec(),
                        shape,
                    },
                ));
            };
        let d = self.cfg.d_model;
        let v = self.cfg.vocab_size;
        push_f32(&mut out, "embed".into(), &self.embed, vec![v, d]);
        push_f32(&mut out, "norm.weight".into(), &self.norm.weight, vec![d]);
        for (i, blk) in self.blocks.iter().enumerate() {
            let p = format!("blocks.{i}");
            push_f32(&mut out, format!("{p}.n1.weight"), &blk.n1.weight, vec![d]);
            push_f32(&mut out, format!("{p}.n2.weight"), &blk.n2.weight, vec![d]);
            push_f32(
                &mut out,
                format!("{p}.n_slot.weight"),
                &blk.n_slot.weight,
                vec![d],
            );
            push_f32(
                &mut out,
                format!("{p}.scan.w_alpha"),
                &blk.scan.w_alpha,
                vec![d],
            );
            push_f32(
                &mut out,
                format!("{p}.scan.b_alpha"),
                &blk.scan.b_alpha,
                vec![d],
            );
            push_f32(&mut out, format!("{p}.scan.w_i"), &blk.scan.w_i, vec![d]);
            push_f32(&mut out, format!("{p}.scan.b_i"), &blk.scan.b_i, vec![d]);
            if !blk.router.is_empty() {
                push_f32(
                    &mut out,
                    format!("{p}.router"),
                    &blk.router,
                    vec![blk.experts.len(), d],
                );
            }
            for (ei, e) in blk.experts.iter().enumerate() {
                let ep = format!("{p}.expert.{ei}");
                let packed_or = |codes: &Option<Vec<f32>>, w: &[f32], rows: usize, cols: usize| {
                    if let Some(c) = codes {
                        NamedBlob::Packed {
                            bytes: pack_ternary(&codes_to_i8(c)),
                            shape: vec![rows, cols],
                        }
                    } else {
                        NamedBlob::F32 {
                            data: w.to_vec(),
                            shape: vec![rows, cols],
                        }
                    }
                };
                out.push((
                    format!("{ep}.w_up"),
                    packed_or(&e.codes_up, &e.w_up, e.w, e.d),
                ));
                out.push((
                    format!("{ep}.w_gate"),
                    packed_or(&e.codes_gate, &e.w_gate, e.w, e.d),
                ));
                out.push((
                    format!("{ep}.w_down"),
                    packed_or(&e.codes_down, &e.w_down, e.d, e.w),
                ));
                push_f32(&mut out, format!("{ep}.bumps"), &e.bumps, vec![e.w, 4]);
                push_f32(&mut out, format!("{ep}.scale_up"), &e.scale_up, vec![e.w]);
                push_f32(
                    &mut out,
                    format!("{ep}.scale_gate"),
                    &e.scale_gate,
                    vec![e.w],
                );
                push_f32(
                    &mut out,
                    format!("{ep}.scale_down"),
                    &e.scale_down,
                    vec![e.d],
                );
            }
            if let Some(s) = &blk.slots {
                push_f32(&mut out, format!("{p}.slots.w_q"), &s.w_q, vec![d]);
                push_f32(&mut out, format!("{p}.slots.w_w"), &s.w_w, vec![d]);
                push_f32(&mut out, format!("{p}.slots.w_link"), &s.w_link, vec![d]);
                push_f32(&mut out, format!("{p}.slots.b_link"), &[s.b_link], vec![1]);
                push_f32(&mut out, format!("{p}.slots.w_bk"), &[s.w_bk], vec![1]);
                push_f32(&mut out, format!("{p}.slots.b_bk"), &[s.b_bk], vec![1]);
                push_f32(&mut out, format!("{p}.slots.w_bv"), &[s.w_bv], vec![1]);
                push_f32(&mut out, format!("{p}.slots.b_bv"), &[s.b_bv], vec![1]);
                push_f32(&mut out, format!("{p}.slots.gamma"), &[s.gamma], vec![1]);
                push_f32(
                    &mut out,
                    format!("{p}.slots.b_alloc"),
                    &[s.b_alloc],
                    vec![1],
                );
            }
        }
        out
    }

    pub fn load_f32(&mut self, name: &str, data: &[f32]) -> Result<()> {
        let copy = |dst: &mut [f32], src: &[f32], n: &str| -> Result<()> {
            if dst.len() != src.len() {
                bail!("{n} len {} != {}", src.len(), dst.len());
            }
            dst.copy_from_slice(src);
            Ok(())
        };
        if name == "embed" {
            return copy(&mut self.embed, data, name);
        }
        if name == "norm.weight" {
            return copy(&mut self.norm.weight, data, name);
        }
        for (i, blk) in self.blocks.iter_mut().enumerate() {
            let p = format!("blocks.{i}");
            if name == format!("{p}.n1.weight") {
                return copy(&mut blk.n1.weight, data, name);
            }
            if name == format!("{p}.n2.weight") {
                return copy(&mut blk.n2.weight, data, name);
            }
            if name == format!("{p}.n_slot.weight") {
                return copy(&mut blk.n_slot.weight, data, name);
            }
            if name == format!("{p}.scan.w_alpha") {
                return copy(&mut blk.scan.w_alpha, data, name);
            }
            if name == format!("{p}.scan.b_alpha") {
                return copy(&mut blk.scan.b_alpha, data, name);
            }
            if name == format!("{p}.scan.w_i") {
                return copy(&mut blk.scan.w_i, data, name);
            }
            if name == format!("{p}.scan.b_i") {
                return copy(&mut blk.scan.b_i, data, name);
            }
            if name == format!("{p}.router") {
                return copy(&mut blk.router, data, name);
            }
            for (ei, e) in blk.experts.iter_mut().enumerate() {
                let ep = format!("{p}.expert.{ei}");
                if name == format!("{ep}.w_up") {
                    return copy(&mut e.w_up, data, name);
                }
                if name == format!("{ep}.w_gate") {
                    return copy(&mut e.w_gate, data, name);
                }
                if name == format!("{ep}.w_down") {
                    return copy(&mut e.w_down, data, name);
                }
                if name == format!("{ep}.bumps") {
                    return copy(&mut e.bumps, data, name);
                }
                if name == format!("{ep}.scale_up") {
                    return copy(&mut e.scale_up, data, name);
                }
                if name == format!("{ep}.scale_gate") {
                    return copy(&mut e.scale_gate, data, name);
                }
                if name == format!("{ep}.scale_down") {
                    return copy(&mut e.scale_down, data, name);
                }
            }
            if let Some(s) = blk.slots.as_mut() {
                if name == format!("{p}.slots.w_q") {
                    return copy(&mut s.w_q, data, name);
                }
                if name == format!("{p}.slots.w_w") {
                    return copy(&mut s.w_w, data, name);
                }
                if name == format!("{p}.slots.w_link") {
                    return copy(&mut s.w_link, data, name);
                }
                if name == format!("{p}.slots.b_link") && !data.is_empty() {
                    s.b_link = data[0];
                    return Ok(());
                }
                if name == format!("{p}.slots.w_bk") && !data.is_empty() {
                    s.w_bk = data[0];
                    return Ok(());
                }
                if name == format!("{p}.slots.b_bk") && !data.is_empty() {
                    s.b_bk = data[0];
                    return Ok(());
                }
                if name == format!("{p}.slots.w_bv") && !data.is_empty() {
                    s.w_bv = data[0];
                    return Ok(());
                }
                if name == format!("{p}.slots.b_bv") && !data.is_empty() {
                    s.b_bv = data[0];
                    return Ok(());
                }
                if name == format!("{p}.slots.gamma") && !data.is_empty() {
                    s.gamma = data[0];
                    return Ok(());
                }
                if name == format!("{p}.slots.b_alloc") && !data.is_empty() {
                    s.b_alloc = data[0];
                    return Ok(());
                }
            }
        }
        Ok(())
    }

    pub fn load_packed(&mut self, name: &str, codes: &[i8]) -> Result<()> {
        let f: Vec<f32> = codes.iter().map(|&c| c as f32).collect();
        for (i, blk) in self.blocks.iter_mut().enumerate() {
            for (ei, e) in blk.experts.iter_mut().enumerate() {
                let ep = format!("blocks.{i}.expert.{ei}");
                if name == format!("{ep}.w_up") {
                    e.codes_up = Some(f.clone());
                    e.packed = true;
                    return Ok(());
                }
                if name == format!("{ep}.w_gate") {
                    e.codes_gate = Some(f.clone());
                    e.packed = true;
                    return Ok(());
                }
                if name == format!("{ep}.w_down") {
                    e.codes_down = Some(f);
                    e.packed = true;
                    return Ok(());
                }
            }
        }
        Ok(())
    }
}

fn ban_recent(logits: &mut [f32], ctx: &[u32]) {
    if ctx.len() >= 4 {
        let last = ctx[ctx.len() - 1];
        if ctx[ctx.len() - 4..].iter().all(|&t| t == last) {
            let i = last as usize;
            if i < logits.len() {
                logits[i] = f32::NEG_INFINITY;
            }
        }
    }
    for &id in ctx.iter().rev().take(16) {
        let i = id as usize;
        if i < logits.len() && logits[i].is_finite() {
            logits[i] -= 1.25;
        }
    }
}

fn upd(opt: &mut DenseSgd, i: &mut usize, w: &mut [f32], g: &[f32], scale: f32) {
    opt.update_slice(*i, w, g, scale);
    *i += 1;
}

fn upd_s(opt: &mut DenseSgd, i: &mut usize, w: &mut f32, g: f32, scale: f32) {
    let mut wv = [*w];
    let gv = [g];
    opt.update_slice(*i, &mut wv, &gv, scale);
    *w = wv[0];
    *i += 1;
}

/// SGD over memory params. Field-by-field so the crate `unsafe_code = deny` holds.
pub fn memory_sgd_step(model: &mut UllisMemory, opt: &mut DenseSgd) -> Result<()> {
    let scale = opt.clip_scale(model.grad_sq());
    let mut i = 0usize;
    let phase = model.phase;
    upd(opt, &mut i, &mut model.embed, &model.embed_grad, scale);
    upd(opt, &mut i, &mut model.norm.weight, &model.norm.grad, scale);
    for blk in &mut model.blocks {
        upd(opt, &mut i, &mut blk.n1.weight, &blk.n1.grad, scale);
        upd(opt, &mut i, &mut blk.n2.weight, &blk.n2.grad, scale);
        upd(opt, &mut i, &mut blk.n_slot.weight, &blk.n_slot.grad, scale);
        upd(
            opt,
            &mut i,
            &mut blk.scan.w_alpha,
            &blk.scan.grad_w_alpha,
            scale,
        );
        upd(
            opt,
            &mut i,
            &mut blk.scan.b_alpha,
            &blk.scan.grad_b_alpha,
            scale,
        );
        upd(opt, &mut i, &mut blk.scan.w_i, &blk.scan.grad_w_i, scale);
        upd(opt, &mut i, &mut blk.scan.b_i, &blk.scan.grad_b_i, scale);
        if !blk.router.is_empty() {
            upd(opt, &mut i, &mut blk.router, &blk.grad_router, scale);
        }
        for e in &mut blk.experts {
            if phase < 4 {
                upd(opt, &mut i, &mut e.w_up, &e.grad_up, scale);
                upd(opt, &mut i, &mut e.w_gate, &e.grad_gate, scale);
                upd(opt, &mut i, &mut e.w_down, &e.grad_down, scale);
                upd(opt, &mut i, &mut e.bumps, &e.grad_bumps, scale);
            }
            upd(opt, &mut i, &mut e.scale_up, &e.grad_scale_up, scale);
            upd(opt, &mut i, &mut e.scale_gate, &e.grad_scale_gate, scale);
            upd(opt, &mut i, &mut e.scale_down, &e.grad_scale_down, scale);
        }
        if let Some(s) = blk.slots.as_mut() {
            upd(opt, &mut i, &mut s.w_q, &s.grad_w_q, scale);
            upd(opt, &mut i, &mut s.w_w, &s.grad_w_w, scale);
            upd(opt, &mut i, &mut s.w_link, &s.grad_w_link, scale);
            upd_s(opt, &mut i, &mut s.b_link, s.grad_b_link, scale);
            upd_s(opt, &mut i, &mut s.w_bk, s.grad_w_bk, scale);
            upd_s(opt, &mut i, &mut s.b_bk, s.grad_b_bk, scale);
            upd_s(opt, &mut i, &mut s.w_bv, s.grad_w_bv, scale);
            upd_s(opt, &mut i, &mut s.b_bv, s.grad_b_bv, scale);
            upd_s(opt, &mut i, &mut s.gamma, s.grad_gamma, scale);
            upd_s(opt, &mut i, &mut s.b_alloc, s.grad_b_alloc, scale);
        }
    }
    Ok(())
}
