//! Autoregressive Small Language Model implementation.
//!
//! Provides two execution engines:
//! 1. `MonolithicSLM`: Standard architecture where all layer weights are fully
//!    materialized in heap memory for the duration of the program.
//! 2. `EphemeralSLM`: Ephemeral streaming architecture where each layer is decrypted from
//!    the sealed payload, dequantized into transient `SecureBuffer`s, executed, and
//!    immediately scrubbed using volatile writes with compiler memory barriers.
//!
//! Both engines read the model from the sealed (encrypted) payload embedded in the
//! executable and need the key to do so. The decrypted INT8 plaintext of a section lives in
//! its own `SecureBuffer` that is wiped as soon as the section has been dequantized.

use crate::ephemeral::{volatile_zero, EphemeralMetrics, SecureBuffer};
use crate::seal::{SealError, SealedPayload, SecretKey};
use crate::telemetry::{get_process_rss_bytes, LifecycleEvent, TelemetryLogger};
use crate::weights::{
    parse_embedding_section, parse_final_section, parse_layer_section, Fp32Slice, ModelConfig,
    QuantizedSlice, EMBEDDED_SEALED_PAYLOAD,
};
use std::time::Instant;

fn dequantized(t: &QuantizedSlice) -> Vec<f32> {
    let mut v = vec![0.0_f32; t.num_elements()];
    t.dequantize_into(&mut v);
    v
}

fn decoded(t: &Fp32Slice) -> Vec<f32> {
    let mut v = vec![0.0_f32; t.len()];
    t.copy_into(&mut v);
    v
}

/// Mathematical primitives for micro-transformer inference.
pub struct MathOps;

impl MathOps {
    /// In-place Root Mean Square Normalization (RMSNorm).
    pub fn rms_norm(x: &mut [f32], gamma: &[f32], eps: f32) {
        let dim = x.len();
        assert_eq!(dim, gamma.len());
        let sum_sq: f32 = x.iter().map(|&v| v * v).sum();
        let rms = (sum_sq / dim as f32 + eps).sqrt();
        let inv_rms = 1.0 / rms;
        for i in 0..dim {
            x[i] = (x[i] * inv_rms) * gamma[i];
        }
    }

    /// Matrix-vector multiplication: y = W * x.
    /// W is [out_dim, in_dim], x is [in_dim], y is [out_dim].
    #[inline]
    pub fn matvec(w: &[f32], x: &[f32], y: &mut [f32], in_dim: usize, out_dim: usize) {
        assert_eq!(w.len(), out_dim * in_dim);
        assert_eq!(x.len(), in_dim);
        assert_eq!(y.len(), out_dim);

        for (i, y_elem) in y.iter_mut().enumerate().take(out_dim) {
            let row_offset = i * in_dim;
            let mut acc = 0.0_f32;
            for j in 0..in_dim {
                acc += w[row_offset + j] * x[j];
            }
            *y_elem = acc;
        }
    }

    /// SiLU (Swish) activation function: f(x) = x / (1 + exp(-x)).
    #[inline]
    pub fn silu_inplace(x: &mut [f32]) {
        for v in x.iter_mut() {
            *v = *v / (1.0 + (-*v).exp());
        }
    }

    /// Multi-head causal self-attention forward pass for a sequence of activations.
    /// `x` has shape [seq_len, d_model].
    #[allow(clippy::too_many_arguments)]
    pub fn multi_head_attention(
        x: &mut [f32],
        seq_len: usize,
        d_model: usize,
        n_heads: usize,
        wq: &[f32],
        wk: &[f32],
        wv: &[f32],
        wo: &[f32],
    ) {
        let head_dim = d_model / n_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Compute Q, K, V for each position
        // Scratch buffers derive directly from the weights and the input, so they are
        // `SecureBuffer`s: they are volatile-zeroed when they go out of scope.
        let mut q = SecureBuffer::<f32>::zeroed(seq_len * d_model);
        let mut k = SecureBuffer::<f32>::zeroed(seq_len * d_model);
        let mut v = SecureBuffer::<f32>::zeroed(seq_len * d_model);

        for t in 0..seq_len {
            let offset = t * d_model;
            let xt = &x[offset..offset + d_model];
            Self::matvec(wq, xt, &mut q[offset..offset + d_model], d_model, d_model);
            Self::matvec(wk, xt, &mut k[offset..offset + d_model], d_model, d_model);
            Self::matvec(wv, xt, &mut v[offset..offset + d_model], d_model, d_model);
        }

        // Output buffer for multi-head attention before final projection
        let mut attn_out = SecureBuffer::<f32>::zeroed(seq_len * d_model);

        // For each attention head
        for h in 0..n_heads {
            let h_offset = h * head_dim;

            // Compute causal attention for each query position t
            for t in 0..seq_len {
                let q_t_offset = t * d_model + h_offset;
                let q_vec = &q[q_t_offset..q_t_offset + head_dim];

                // Compute scores against all prior and current positions s <= t
                let mut scores = SecureBuffer::<f32>::zeroed(t + 1);
                let mut max_score = f32::NEG_INFINITY;

                for s in 0..=t {
                    let k_s_offset = s * d_model + h_offset;
                    let k_vec = &k[k_s_offset..k_s_offset + head_dim];
                    let mut dot = 0.0_f32;
                    for d in 0..head_dim {
                        dot += q_vec[d] * k_vec[d];
                    }
                    let score = dot * scale;
                    if score > max_score {
                        max_score = score;
                    }
                    scores[s] = score;
                }

                // Causal softmax
                let mut exp_sum = 0.0_f32;
                for sc in scores.iter_mut() {
                    *sc = (*sc - max_score).exp();
                    exp_sum += *sc;
                }
                let inv_sum = 1.0 / exp_sum;
                for sc in scores.iter_mut() {
                    *sc *= inv_sum;
                }

                // Weighted sum over V
                let out_offset = t * d_model + h_offset;
                for d in 0..head_dim {
                    let mut acc = 0.0_f32;
                    for (s, &sc) in scores.iter().enumerate().take(t + 1) {
                        let v_s_offset = s * d_model + h_offset;
                        acc += sc * v[v_s_offset + d];
                    }
                    attn_out[out_offset + d] = acc;
                }
            }
        }

        // Final linear projection and residual addition: x = x + wo * attn_out
        let mut proj = SecureBuffer::<f32>::zeroed(d_model);
        for t in 0..seq_len {
            let offset = t * d_model;
            Self::matvec(wo, &attn_out[offset..offset + d_model], &mut proj, d_model, d_model);
            for d in 0..d_model {
                x[offset + d] += proj[d];
            }
        }
    }

    /// FeedForward MLP block: x = x + w2 * silu(w1 * rmsnorm(x)).
    pub fn feed_forward(
        x: &mut [f32],
        seq_len: usize,
        d_model: usize,
        d_ff: usize,
        norm2: &[f32],
        w1: &[f32],
        w2: &[f32],
    ) {
        let mut h = SecureBuffer::<f32>::zeroed(d_model);
        let mut hidden = SecureBuffer::<f32>::zeroed(d_ff);
        let mut out = SecureBuffer::<f32>::zeroed(d_model);

        for t in 0..seq_len {
            let offset = t * d_model;
            h.copy_from_slice(&x[offset..offset + d_model]);
            Self::rms_norm(&mut h, norm2, 1e-5);

            // Linear 1: d_model -> d_ff
            Self::matvec(w1, &h, &mut hidden, d_model, d_ff);
            Self::silu_inplace(&mut hidden);

            // Linear 2: d_ff -> d_model
            Self::matvec(w2, &hidden, &mut out, d_ff, d_model);

            // Residual addition
            for d in 0..d_model {
                x[offset + d] += out[d];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// MONOLITHIC SLM: Weights retained in RAM throughout execution
// ---------------------------------------------------------------------------

pub struct MonolithicLayer {
    pub norm1: Vec<f32>,
    pub wq: Vec<f32>,
    pub wk: Vec<f32>,
    pub wv: Vec<f32>,
    pub wo: Vec<f32>,
    pub norm2: Vec<f32>,
    pub w1: Vec<f32>,
    pub w2: Vec<f32>,
}

pub struct MonolithicSLM {
    pub config: ModelConfig,
    pub tok_emb: Vec<f32>,
    pub pos_emb: Vec<f32>,
    pub layers: Vec<MonolithicLayer>,
    pub final_norm: Vec<f32>,
    pub head: Vec<f32>,
    pub total_param_bytes: usize,
    /// Decrypted INT8 plaintext that was wiped during construction.
    pub plaintext_bytes_wiped: usize,
}

impl MonolithicSLM {
    /// Build the baseline from the sealed payload embedded in the executable.
    pub fn open(key: &SecretKey) -> Result<Self, SealError> {
        Self::from_sealed(EMBEDDED_SEALED_PAYLOAD, key)
    }

    /// Decrypt every section once and keep all dequantized weights resident (the baseline).
    ///
    /// Each section is decrypted into its own `SecureBuffer` that is volatile-wiped as soon
    /// as it has been dequantized, so the INT8 plaintext does not outlive this call.
    pub fn from_sealed(raw: &[u8], key: &SecretKey) -> Result<Self, SealError> {
        let sealed = SealedPayload::parse(raw)?;
        let cfg = sealed.config;
        let mut plaintext_bytes_wiped = 0;

        let open = |i: usize| -> Result<SecureBuffer<u8>, SealError> {
            let mut plain = SecureBuffer::<u8>::zeroed(sealed.section_plain_len(i));
            sealed.open_section(i, key, &mut plain)?;
            Ok(plain)
        };

        let mut plain = open(0)?;
        let (tok_emb, pos_emb) = {
            let (tok, pos) = parse_embedding_section(&plain);
            (dequantized(&tok), dequantized(&pos))
        };
        plaintext_bytes_wiped += plain.size_bytes();
        plain.scrub();
        let mut total_param_bytes = (tok_emb.len() + pos_emb.len()) * 4;

        let mut layers = Vec::with_capacity(cfg.n_layers);
        for l in 0..cfg.n_layers {
            let mut plain = open(1 + l)?;
            let layer = {
                let ls = parse_layer_section(&plain);
                MonolithicLayer {
                    norm1: decoded(&ls.norm1),
                    wq: dequantized(&ls.wq),
                    wk: dequantized(&ls.wk),
                    wv: dequantized(&ls.wv),
                    wo: dequantized(&ls.wo),
                    norm2: decoded(&ls.norm2),
                    w1: dequantized(&ls.w1),
                    w2: dequantized(&ls.w2),
                }
            };
            plaintext_bytes_wiped += plain.size_bytes();
            plain.scrub();
            total_param_bytes += (layer.norm1.len()
                + layer.wq.len()
                + layer.wk.len()
                + layer.wv.len()
                + layer.wo.len()
                + layer.norm2.len()
                + layer.w1.len()
                + layer.w2.len())
                * 4;
            layers.push(layer);
        }

        let mut plain = open(cfg.n_layers + 1)?;
        let (final_norm, head) = {
            let (norm, head) = parse_final_section(&plain);
            (decoded(&norm), dequantized(&head))
        };
        plaintext_bytes_wiped += plain.size_bytes();
        plain.scrub();
        total_param_bytes += (final_norm.len() + head.len()) * 4;

        Ok(Self {
            config: cfg,
            tok_emb,
            pos_emb,
            layers,
            final_norm,
            head,
            total_param_bytes,
            plaintext_bytes_wiped,
        })
    }

    /// Compute forward pass and return logits for the final token in the prompt.
    pub fn forward(&self, tokens: &[u8]) -> Vec<f32> {
        let seq_len = tokens.len();
        let d = self.config.d_model;
        let mut x = vec![0.0_f32; seq_len * d];

        // Embedding lookup
        for t in 0..seq_len {
            let tok_idx = tokens[t] as usize;
            let tok_row = &self.tok_emb[tok_idx * d..(tok_idx + 1) * d];
            let pos_row = &self.pos_emb[t * d..(t + 1) * d];
            for j in 0..d {
                x[t * d + j] = tok_row[j] + pos_row[j];
            }
        }

        // Evaluate transformer layers
        for layer in &self.layers {
            // Attention block with pre-norm
            let mut h_norm = x.clone();
            for t in 0..seq_len {
                MathOps::rms_norm(&mut h_norm[t * d..(t + 1) * d], &layer.norm1, 1e-5);
            }
            MathOps::multi_head_attention(
                &mut h_norm,
                seq_len,
                d,
                self.config.n_heads,
                &layer.wq,
                &layer.wk,
                &layer.wv,
                &layer.wo,
            );
            x.copy_from_slice(&h_norm);

            // MLP block with pre-norm
            MathOps::feed_forward(
                &mut x,
                seq_len,
                d,
                self.config.d_ff,
                &layer.norm2,
                &layer.w1,
                &layer.w2,
            );
        }

        // Final RMSNorm on the last token activation
        let last_offset = (seq_len - 1) * d;
        let mut final_act = x[last_offset..last_offset + d].to_vec();
        MathOps::rms_norm(&mut final_act, &self.final_norm, 1e-5);

        // Project with head: logits = head * final_act
        let mut logits = vec![0.0_f32; self.config.vocab_size];
        MathOps::matvec(&self.head, &final_act, &mut logits, d, self.config.vocab_size);
        logits
    }

    /// Naive cleanup: plain `fill(0.0)`, the method a typical program would use.
    ///
    /// The compiler is allowed to drop these stores as dead if the buffers are freed
    /// right afterwards, so this gives no guarantee. Kept so the forensic scan can show
    /// what the naive approach actually leaves behind.
    pub fn cleanup_naive(&mut self) {
        self.tok_emb.fill(0.0);
        self.pos_emb.fill(0.0);
        for l in &mut self.layers {
            l.norm1.fill(0.0);
            l.wq.fill(0.0);
            l.wk.fill(0.0);
            l.wv.fill(0.0);
            l.wo.fill(0.0);
            l.norm2.fill(0.0);
            l.w1.fill(0.0);
            l.w2.fill(0.0);
        }
        self.final_norm.fill(0.0);
        self.head.fill(0.0);
    }

    /// Secure cleanup: volatile zeroization with compiler fences (the same primitive the
    /// ephemeral engine uses), so the two engines are compared with the same wipe.
    pub fn cleanup(&mut self) {
        volatile_zero(&mut self.tok_emb);
        volatile_zero(&mut self.pos_emb);
        for l in &mut self.layers {
            volatile_zero(&mut l.norm1);
            volatile_zero(&mut l.wq);
            volatile_zero(&mut l.wk);
            volatile_zero(&mut l.wv);
            volatile_zero(&mut l.wo);
            volatile_zero(&mut l.norm2);
            volatile_zero(&mut l.w1);
            volatile_zero(&mut l.w2);
        }
        volatile_zero(&mut self.final_norm);
        volatile_zero(&mut self.head);
    }
}

// ---------------------------------------------------------------------------
// EPHEMERAL SLM: Transient layer streaming & volatile scrubbing
// ---------------------------------------------------------------------------

/// Transient buffer holding reconstructed weights for a single layer during its forward pass.
pub struct TransientLayerBuffer {
    pub norm1: SecureBuffer<f32>,
    pub wq: SecureBuffer<f32>,
    pub wk: SecureBuffer<f32>,
    pub wv: SecureBuffer<f32>,
    pub wo: SecureBuffer<f32>,
    pub norm2: SecureBuffer<f32>,
    pub w1: SecureBuffer<f32>,
    pub w2: SecureBuffer<f32>,
}

impl TransientLayerBuffer {
    pub fn total_bytes(&self) -> usize {
        self.norm1.size_bytes()
            + self.wq.size_bytes()
            + self.wk.size_bytes()
            + self.wv.size_bytes()
            + self.wo.size_bytes()
            + self.norm2.size_bytes()
            + self.w1.size_bytes()
            + self.w2.size_bytes()
    }

    pub fn scrub_all(&mut self) {
        self.norm1.scrub();
        self.wq.scrub();
        self.wk.scrub();
        self.wv.scrub();
        self.wo.scrub();
        self.norm2.scrub();
        self.w1.scrub();
        self.w2.scrub();
    }

    pub fn audit_all_zeroed(&self) -> bool {
        self.norm1.audit_is_zero()
            && self.wq.audit_is_zero()
            && self.wk.audit_is_zero()
            && self.wv.audit_is_zero()
            && self.wo.audit_is_zero()
            && self.norm2.audit_is_zero()
            && self.w1.audit_is_zero()
            && self.w2.audit_is_zero()
    }
}


pub struct EphemeralSLM<'a> {
    sealed: SealedPayload<'a>,
    key: &'a SecretKey,
    pub metrics: EphemeralMetrics,
}

impl<'a> EphemeralSLM<'a> {
    /// Use the sealed payload embedded in the executable.
    pub fn open(key: &'a SecretKey) -> Result<Self, SealError> {
        Self::from_sealed(EMBEDDED_SEALED_PAYLOAD, key)
    }

    /// Use any sealed payload. Every section's authentication tag is checked up front
    /// (no plaintext is produced), so a wrong key or a damaged file is rejected here,
    /// before any work is done. Each section is checked again when it is decrypted.
    pub fn from_sealed(raw: &'a [u8], key: &'a SecretKey) -> Result<Self, SealError> {
        let sealed = SealedPayload::parse(raw)?;
        sealed.verify_all(key)?;
        Ok(Self {
            sealed,
            key,
            metrics: EphemeralMetrics::default(),
        })
    }

    pub fn config(&self) -> ModelConfig {
        self.sealed.config
    }

    /// Size of the sealed payload embedded in the executable.
    pub fn payload_bytes(&self) -> usize {
        self.sealed.total_bytes()
    }

    /// Decrypt section `i` into a fresh transient buffer. Nothing is written if the tag
    /// does not verify.
    fn open_section(&self, i: usize) -> Result<SecureBuffer<u8>, SealError> {
        let mut plain = SecureBuffer::<u8>::zeroed(self.sealed.section_plain_len(i));
        self.sealed.open_section(i, self.key, &mut plain)?;
        Ok(plain)
    }

    /// Wipe a decrypted section, read it back, and record the result.
    fn wipe_plaintext(metrics: &mut EphemeralMetrics, mut plain: SecureBuffer<u8>) {
        let bytes = plain.size_bytes();
        plain.scrub();
        let clean = plain.audit_is_zero();
        metrics.record_plaintext_wipe(bytes, clean);
    }

    /// Forward pass where each section is decrypted, dequantized, executed, and wiped in
    /// turn. The decrypted INT8 plaintext is wiped as soon as it has been dequantized (before
    /// the layer runs); the dequantized f32 weights are wiped right after the layer runs.
    pub fn forward_ephemeral(
        &mut self,
        tokens: &[u8],
        telemetry: &TelemetryLogger,
    ) -> Result<Vec<f32>, SealError> {
        let seq_len = tokens.len();
        let cfg = self.sealed.config;
        let d = cfg.d_model;
        let mut x = SecureBuffer::<f32>::zeroed(seq_len * d);

        // 1. Transient Embedding Lookup:
        // Materialize embeddings only for the active tokens, then scrub buffer.
        {
            let plain = self.open_section(0)?;
            let (tok_slice, pos_slice) = parse_embedding_section(&plain);
            let mut transient_tok = SecureBuffer::zeroed(tok_slice.num_elements());
            let mut transient_pos = SecureBuffer::zeroed(pos_slice.num_elements());
            let emb_bytes = transient_tok.size_bytes() + transient_pos.size_bytes();
            self.metrics.record_allocation(emb_bytes);

            tok_slice.dequantize_into(&mut transient_tok);
            pos_slice.dequantize_into(&mut transient_pos);
            Self::wipe_plaintext(&mut self.metrics, plain);

            for t in 0..seq_len {
                let tok_idx = tokens[t] as usize;
                let tok_row = &transient_tok[tok_idx * d..(tok_idx + 1) * d];
                let pos_row = &transient_pos[t * d..(t + 1) * d];
                for j in 0..d {
                    x[t * d + j] = tok_row[j] + pos_row[j];
                }
            }

            // Immediately scrub embedding buffers
            transient_tok.scrub();
            transient_pos.scrub();
            let passed = transient_tok.audit_is_zero() && transient_pos.audit_is_zero();
            self.metrics.record_scrub(emb_bytes);
            self.metrics.record_audit(passed);
        }

        // 2. Transient Layer Streaming:
        // Each layer is decrypted, dequantized, evaluated, and immediately scrubbed.
        for layer_idx in 0..cfg.n_layers {
            let reconstruct_start = Instant::now();
            let plain = self.open_section(1 + layer_idx)?;
            let slices = parse_layer_section(&plain);

            let mut norm1 = SecureBuffer::zeroed(slices.norm1.len());
            slices.norm1.copy_into(&mut norm1);

            let mut wq = SecureBuffer::zeroed(slices.wq.num_elements());
            slices.wq.dequantize_into(&mut wq);

            let mut wk = SecureBuffer::zeroed(slices.wk.num_elements());
            slices.wk.dequantize_into(&mut wk);

            let mut wv = SecureBuffer::zeroed(slices.wv.num_elements());
            slices.wv.dequantize_into(&mut wv);

            let mut wo = SecureBuffer::zeroed(slices.wo.num_elements());
            slices.wo.dequantize_into(&mut wo);

            let mut norm2 = SecureBuffer::zeroed(slices.norm2.len());
            slices.norm2.copy_into(&mut norm2);

            let mut w1 = SecureBuffer::zeroed(slices.w1.num_elements());
            slices.w1.dequantize_into(&mut w1);

            let mut w2 = SecureBuffer::zeroed(slices.w2.num_elements());
            slices.w2.dequantize_into(&mut w2);

            // The INT8 plaintext is no longer needed: wipe it before computing.
            Self::wipe_plaintext(&mut self.metrics, plain);

            let mut layer_buf = TransientLayerBuffer {
                norm1,
                wq,
                wk,
                wv,
                wo,
                norm2,
                w1,
                w2,
            };

            let layer_bytes = layer_buf.total_bytes();
            self.metrics.record_allocation(layer_bytes);
            let recon_us = reconstruct_start.elapsed().as_micros();

            telemetry.log(LifecycleEvent::LayerReconstructed {
                layer_idx,
                tensor_bytes: layer_bytes,
                elapsed_us: recon_us,
                rss_kb: get_process_rss_bytes() / 1024,
            });

            // Run attention with pre-norm
            let mut h_norm = SecureBuffer::from_vec(x.to_vec());
            for t in 0..seq_len {
                MathOps::rms_norm(&mut h_norm[t * d..(t + 1) * d], &layer_buf.norm1, 1e-5);
            }
            MathOps::multi_head_attention(
                &mut h_norm,
                seq_len,
                d,
                cfg.n_heads,
                &layer_buf.wq,
                &layer_buf.wk,
                &layer_buf.wv,
                &layer_buf.wo,
            );
            x.copy_from_slice(&h_norm);

            // Run MLP with pre-norm
            MathOps::feed_forward(
                &mut x,
                seq_len,
                d,
                cfg.d_ff,
                &layer_buf.norm2,
                &layer_buf.w1,
                &layer_buf.w2,
            );

            // Secure volatile scrub of active layer buffer
            layer_buf.scrub_all();
            let audit_ok = layer_buf.audit_all_zeroed();
            self.metrics.record_scrub(layer_bytes);
            self.metrics.record_audit(audit_ok);

            telemetry.log(LifecycleEvent::LayerScrubbed {
                layer_idx,
                scrubbed_bytes: layer_bytes,
                audit_zeroed: audit_ok,
                rss_kb: get_process_rss_bytes() / 1024,
            });
        }

        // 3. Transient Head Evaluation:
        let plain = self.open_section(cfg.n_layers + 1)?;
        let (fn_slice, head_slice) = parse_final_section(&plain);
        let mut final_norm = SecureBuffer::zeroed(fn_slice.len());
        fn_slice.copy_into(&mut final_norm);

        let mut head = SecureBuffer::zeroed(head_slice.num_elements());
        head_slice.dequantize_into(&mut head);
        Self::wipe_plaintext(&mut self.metrics, plain);

        let head_bytes = final_norm.size_bytes() + head.size_bytes();
        self.metrics.record_allocation(head_bytes);

        // Normalize last token
        let last_offset = (seq_len - 1) * d;
        let mut final_act = SecureBuffer::from_vec(x[last_offset..last_offset + d].to_vec());
        MathOps::rms_norm(&mut final_act, &final_norm, 1e-5);

        let mut logits = vec![0.0_f32; cfg.vocab_size];
        MathOps::matvec(&head, &final_act, &mut logits, d, cfg.vocab_size);

        // Scrub head buffers
        final_norm.scrub();
        head.scrub();
        let passed = final_norm.audit_is_zero() && head.audit_is_zero();
        self.metrics.record_scrub(head_bytes);
        self.metrics.record_audit(passed);

        Ok(logits)
    }
}
