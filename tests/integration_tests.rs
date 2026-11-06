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
