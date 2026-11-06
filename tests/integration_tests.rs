use ephemeral_slm_research::{
    ephemeral::SecureBuffer,
    model::{EphemeralSLM, MonolithicSLM},
    telemetry::TelemetryLogger,
    weights::{PayloadView, EMBEDDED_MODEL_PAYLOAD},
};

#[test]
fn test_payload_view_parses_cleanly() {
    let payload = PayloadView::parse(EMBEDDED_MODEL_PAYLOAD).expect("Payload parse failed");
    let cfg = payload.config;
    assert_eq!(cfg.version, 1);
    assert_eq!(cfg.vocab_size, 128);
    assert_eq!(cfg.ctx_len, 32);
    assert_eq!(cfg.d_model, 48);
    assert_eq!(cfg.n_layers, 2);
    assert_eq!(cfg.n_heads, 3);
    assert_eq!(cfg.d_ff, 96);
}

#[test]
fn test_secure_buffer_scrubbing_and_audit() {
    let mut buf: SecureBuffer<f32> = SecureBuffer::zeroed(64);
    assert!(buf.audit_is_zero());

    // Populate with non-zero weights
    for (i, v) in buf.iter_mut().enumerate() {
        *v = (i + 1) as f32 * 0.25;
    }
    assert!(!buf.audit_is_zero());
    assert!(!buf.is_scrubbed());

    // Perform secure volatile scrub
    buf.scrub();
    assert!(buf.is_scrubbed());
    assert!(buf.audit_is_zero());

    // Verify all bytes are 0
    let slice = &buf[..];
    for &val in slice {
        assert_eq!(val, 0.0);
    }
}

#[test]
fn test_monolithic_vs_ephemeral_parity() {
    let mono = MonolithicSLM::new();
    let mut ephem = EphemeralSLM::new();
    let telemetry = TelemetryLogger::new(false);

    let prompt = b"The ephemeral ";
    let mono_logits = mono.forward(prompt);
    let ephem_logits = ephem.forward_ephemeral(prompt, &telemetry);

    assert_eq!(mono_logits.len(), ephem_logits.len());
    assert_eq!(mono_logits.len(), 128);

    // Verify mathematical parity between monolithic and ephemeral execution
    for i in 0..mono_logits.len() {
        let diff = (mono_logits[i] - ephem_logits[i]).abs();
        assert!(
            diff < 1e-4,
            "Mismatch at logit index {}: mono={}, ephem={}, diff={}",
            i,
            mono_logits[i],
            ephem_logits[i],
            diff
        );
    }

    // Verify ephemeral audits passed
    assert!(ephem.metrics.audits_passed > 0);
    assert_eq!(ephem.metrics.audits_failed, 0);
    assert_eq!(ephem.metrics.current_active_bytes, 0);
}

#[test]
fn test_volatile_zero_clears_any_copy_type() {
    use ephemeral_slm_research::volatile_zero;

    let mut floats = [1.5_f32, -2.25, 3.0, 4.75];
    volatile_zero(&mut floats);
    assert!(floats.iter().all(|&v| v == 0.0));

    let mut words = vec![0xDEAD_BEEF_u32; 33];
    volatile_zero(&mut words);
    assert!(words.iter().all(|&v| v == 0));

    let mut empty: [u8; 0] = [];
    volatile_zero(&mut empty); // must not panic
}

/// Regression test: `include_bytes!` data is only guaranteed 1-byte aligned. The payload
/// must decode identically (and without undefined behaviour) at every base alignment.
#[test]
fn test_payload_decodes_identically_at_every_alignment() {
    fn decode_all(raw: &[u8]) -> Vec<f32> {
        let payload = PayloadView::parse(raw).expect("payload parse failed");
        let mut out = Vec::new();
        let (tok, _pos) = payload.get_embeddings();
        let mut buf = vec![0.0_f32; tok.num_elements()];
        tok.dequantize_into(&mut buf);
        out.extend_from_slice(&buf[..16]);
        for l in 0..payload.config.n_layers {
            let s = payload.get_layer_slices(l);
            let mut n1 = vec![0.0_f32; s.norm1.len()];
            s.norm1.copy_into(&mut n1);
            let mut n2 = vec![0.0_f32; s.norm2.len()];
            s.norm2.copy_into(&mut n2);
            out.extend(n1);
            out.extend(n2);
        }
        let (fnorm, _head) = payload.get_final_slices();
        let mut f = vec![0.0_f32; fnorm.len()];
        fnorm.copy_into(&mut f);
        out.extend(f);
        out
    }

    let reference = decode_all(EMBEDDED_MODEL_PAYLOAD);
    assert!(reference.iter().any(|&v| v != 0.0));

    for shift in 1..4 {
        // Copy the payload so that its base pointer sits `shift` bytes past an aligned address.
        let mut backing = vec![0u8; EMBEDDED_MODEL_PAYLOAD.len() + 8];
        let base = backing.as_ptr() as usize;
        let start = (4 - base % 4) % 4 + shift; // 4-byte aligned, then +shift
        backing[start..start + EMBEDDED_MODEL_PAYLOAD.len()].copy_from_slice(EMBEDDED_MODEL_PAYLOAD);
        let misaligned = &backing[start..start + EMBEDDED_MODEL_PAYLOAD.len()];
        assert_ne!(misaligned.as_ptr() as usize % 4, 0, "test setup: pointer must be misaligned");
        assert_eq!(decode_all(misaligned), reference, "decode differs at shift {shift}");
    }
}

#[test]
fn test_monolithic_cleanup_zeroes_every_weight() {
    for secure in [false, true] {
        let mut m = MonolithicSLM::new();
        assert!(m.head.iter().any(|&v| v != 0.0));
        if secure {
            m.cleanup();
        } else {
            m.cleanup_naive();
        }
        let all_zero = m.tok_emb.iter().chain(&m.pos_emb).chain(&m.final_norm).chain(&m.head)
            .chain(m.layers.iter().flat_map(|l| {
                l.norm1.iter().chain(&l.wq).chain(&l.wk).chain(&l.wv).chain(&l.wo)
                    .chain(&l.norm2).chain(&l.w1).chain(&l.w2)
            }))
            .all(|&v| v == 0.0);
        assert!(all_zero, "cleanup (secure={secure}) left non-zero weights");
    }
}

/// End-to-end process-memory scan. Each scenario runs in its own child process; the
/// parent exits non-zero if the positive control finds nothing (scanner broken) or if a
/// scrubbed scenario still contains decompressed weights.
#[test]
fn test_process_memory_forensics() {
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_ephemeral-slm-research"))
        .arg("--forensics")
        .output()
        .expect("failed to run the binary");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "forensics failed:\n{stdout}\n{}", String::from_utf8_lossy(&out.stderr));
    if ephemeral_slm_research::forensics::is_supported() {
        assert!(stdout.contains("all assertions held"), "{stdout}");
    }
}
