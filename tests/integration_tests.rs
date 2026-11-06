use ephemeral_slm_research::{
    ephemeral::SecureBuffer,
    forensics,
    model::{EphemeralSLM, MonolithicSLM},
    seal::{SealError, SealedPayload, SecretKey},
    telemetry::TelemetryLogger,
    weights::{PayloadView, EMBEDDED_SEALED_PAYLOAD},
};
use std::collections::HashMap;

/// The plaintext training output. It is only used here as a reference to compare against;
/// the shipped executable embeds `EMBEDDED_SEALED_PAYLOAD` (ciphertext) instead.
const PLAIN: &[u8] = include_bytes!("../assets/weights.bin");
const KEY_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/assets/demo.key");
const EXE: &str = env!("CARGO_BIN_EXE_ephemeral-slm-research");

fn demo_key() -> SecretKey {
    SecretKey::from_file(KEY_PATH).expect("assets/demo.key must exist")
}

fn wrong_key() -> SecretKey {
    SecretKey::from_bytes([0x42; 32])
}

#[test]
fn test_payload_view_parses_cleanly() {
    let payload = PayloadView::parse(PLAIN).expect("Payload parse failed");
    let cfg = payload.config;
    assert_eq!(cfg.version, 1);
    assert_eq!(cfg.vocab_size, 128);
    assert_eq!(cfg.ctx_len, 32);
    assert_eq!(cfg.d_model, 48);
    assert_eq!(cfg.n_layers, 2);
    assert_eq!(cfg.n_heads, 3);
    assert_eq!(cfg.d_ff, 96);

    // The sealed container advertises the same configuration.
    let sealed = SealedPayload::parse(EMBEDDED_SEALED_PAYLOAD).expect("sealed parse failed");
    assert_eq!(sealed.config, cfg);
    assert_eq!(sealed.section_count(), cfg.n_layers + 2);
}

/// The embedded ciphertext was produced independently (Python `cryptography`), so this
/// also checks that the Rust decryptor interoperates with a vetted implementation.
#[test]
fn test_embedded_payload_is_the_encryption_of_weights_bin() {
    let sealed = SealedPayload::parse(EMBEDDED_SEALED_PAYLOAD).unwrap();
    let opened = sealed.open_all(&demo_key()).expect("demo key must open the embedded payload");
    assert_eq!(opened, PLAIN, "decrypted embedded payload differs from assets/weights.bin");
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
    let key = demo_key();
    let mono = MonolithicSLM::open(&key).unwrap();
    let mut ephem = EphemeralSLM::open(&key).unwrap();
    let telemetry = TelemetryLogger::new(false);

    let prompt = b"The ephemeral ";
    let mono_logits = mono.forward(prompt);
    let ephem_logits = ephem.forward_ephemeral(prompt, &telemetry).unwrap();

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

/// Every section is decrypted exactly once per forward pass, and every byte of decrypted
/// INT8 plaintext is wiped and audited.
#[test]
fn test_ephemeral_decrypts_and_wipes_every_section() {
    let key = demo_key();
    let mut ephem = EphemeralSLM::open(&key).unwrap();
    let telemetry = TelemetryLogger::new(false);
    ephem.forward_ephemeral(b"The ephemeral ", &telemetry).unwrap();
    ephem.forward_ephemeral(b"Safe ", &telemetry).unwrap();

    let sections = SealedPayload::parse(EMBEDDED_SEALED_PAYLOAD).unwrap().section_count();
    let plaintext_total = PLAIN.len() - 32; // everything after the 32-byte plaintext header
    assert_eq!(ephem.metrics.sections_opened, 2 * sections);
    assert_eq!(ephem.metrics.plaintext_bytes_wiped, 2 * plaintext_total);
    assert_eq!(ephem.metrics.audits_failed, 0);

    let mono = MonolithicSLM::open(&key).unwrap();
    assert_eq!(mono.plaintext_bytes_wiped, plaintext_total);
}

#[test]
fn test_wrong_key_is_rejected_by_both_engines() {
    let bad = wrong_key();
    match MonolithicSLM::open(&bad) {
        Err(SealError::AuthFailed { section: 0 }) => {}
        other => panic!("monolithic must reject a wrong key, got {:?}", other.map(|_| "a model")),
    }

    // The ephemeral engine verifies every tag when it is opened, so it also fails up
    // front, before any section has been decrypted.
    match EphemeralSLM::open(&bad) {
        Err(SealError::AuthFailed { section: 0 }) => {}
        other => panic!("ephemeral must reject a wrong key, got {:?}", other.map(|_| "a model")),
    }
}

#[test]
fn test_tampered_payload_is_rejected_at_run_time() {
    let key = demo_key();

    // Flip one bit in the middle of the ciphertext (inside a layer section).
    let mut bad = EMBEDDED_SEALED_PAYLOAD.to_vec();
    let mid = bad.len() / 2;
    bad[mid] ^= 0x01;

    assert!(matches!(
        MonolithicSLM::from_sealed(&bad, &key),
        Err(SealError::AuthFailed { .. })
    ));
    assert!(matches!(
        EphemeralSLM::from_sealed(&bad, &key),
        Err(SealError::AuthFailed { .. })
    ));

    // The pristine payload still works with the same key, in both engines.
    assert!(MonolithicSLM::from_sealed(EMBEDDED_SEALED_PAYLOAD, &key).is_ok());
    assert!(EphemeralSLM::from_sealed(EMBEDDED_SEALED_PAYLOAD, &key).is_ok());
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

/// Copy `data` so that its base pointer sits `shift` bytes past a 4-byte-aligned address.
fn misaligned_copy(data: &[u8], shift: usize) -> (Vec<u8>, usize) {
    let mut backing = vec![0u8; data.len() + 8];
    let base = backing.as_ptr() as usize;
    let start = (4 - base % 4) % 4 + shift; // 4-byte aligned, then +shift
    backing[start..start + data.len()].copy_from_slice(data);
    assert_ne!(backing[start..].as_ptr() as usize % 4, 0, "test setup: pointer must be misaligned");
    (backing, start)
}

/// Regression test: a payload is only guaranteed 1-byte aligned. It must decode identically
/// (and without undefined behaviour) at every base alignment, both the plaintext layout and
/// the sealed container.
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

    let reference = decode_all(PLAIN);
    assert!(reference.iter().any(|&v| v != 0.0));
    let key = demo_key();

    for shift in 1..4 {
        let (backing, start) = misaligned_copy(PLAIN, shift);
        assert_eq!(decode_all(&backing[start..start + PLAIN.len()]), reference, "plaintext differs at shift {shift}");

        let (backing, start) = misaligned_copy(EMBEDDED_SEALED_PAYLOAD, shift);
        let sealed = SealedPayload::parse(&backing[start..start + EMBEDDED_SEALED_PAYLOAD.len()]).unwrap();
        assert_eq!(sealed.open_all(&key).unwrap(), PLAIN, "sealed payload differs at shift {shift}");
    }
}

#[test]
fn test_monolithic_cleanup_zeroes_every_weight() {
    let key = demo_key();
    for secure in [false, true] {
        let mut m = MonolithicSLM::open(&key).unwrap();
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

// ---------------------------------------------------------------------------
// The executable itself
// ---------------------------------------------------------------------------

/// Index 32-byte chunks by their first 8 bytes, then count how many offsets of `haystack`
/// match any chunk. Chunks with few distinct bytes are skipped (zero runs, repeats).
fn count_chunk_hits(haystack: &[u8], source: &[u8]) -> (usize, usize) {
    let mut index: HashMap<u64, Vec<&[u8]>> = HashMap::new();
    let mut chunks = 0;
    for chunk in source.chunks_exact(32) {
        let mut seen = [false; 256];
        for &b in chunk {
            seen[b as usize] = true;
        }
        if seen.iter().filter(|&&s| s).count() < 12 {
            continue;
        }
        chunks += 1;
        index
            .entry(u64::from_le_bytes(chunk[..8].try_into().unwrap()))
            .or_default()
            .push(chunk);
    }
    let mut hits = 0;
    for off in 0..haystack.len().saturating_sub(32) {
        let key = u64::from_le_bytes(haystack[off..off + 8].try_into().unwrap());
        if let Some(candidates) = index.get(&key) {
            if candidates.iter().any(|c| *c == &haystack[off..off + 32]) {
                hits += 1;
            }
        }
    }
    (hits, chunks)
}

/// The point of sealing: the built executable carries ciphertext only. No 32-byte piece of
/// the plaintext payload may appear in it, while pieces of the sealed payload (the control)
/// do, so the search is known to work.
#[test]
fn test_executable_contains_ciphertext_but_no_plaintext_weights() {
    let exe = std::fs::read(EXE).expect("cannot read the built executable");

    let (control_hits, control_chunks) = count_chunk_hits(&exe, &EMBEDDED_SEALED_PAYLOAD[64..]);
    assert!(control_chunks > 1000, "control has too few chunks: {control_chunks}");
    assert!(
        control_hits >= control_chunks * 9 / 10,
        "control failed: only {control_hits} of {control_chunks} sealed chunks found in the executable"
    );

    // Skip the 32-byte plaintext header: it holds the magic and config, which the sealed
    // header also carries in the clear.
    let (plain_hits, plain_chunks) = count_chunk_hits(&exe, &PLAIN[32..]);
    assert!(plain_chunks > 1000, "plaintext has too few chunks: {plain_chunks}");
    assert_eq!(plain_hits, 0, "{plain_hits} plaintext weight chunks found in the executable");

    // And the specific signatures the forensics module searches for.
    for sig in forensics::compute_signatures(&demo_key()).unwrap() {
        assert!(
            !exe.windows(32).any(|w| w == sig.bytes),
            "signature {} found in the executable",
            sig.label
        );
    }
}

// ---------------------------------------------------------------------------
// Command line
// ---------------------------------------------------------------------------

fn temp_path(name: &str) -> std::path::PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("eslm-test-{}-{name}", std::process::id()));
    p
}

#[test]
fn test_cli_requires_a_key() {
    let out = std::process::Command::new(EXE)
        .args(["--tokens", "2"])
        .env_remove("ESLM_KEY")
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(2));
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("--key-file") && err.contains("ESLM_KEY"), "unhelpful error: {err}");
}

#[test]
fn test_cli_wrong_key_fails_cleanly_and_prints_no_text() {
    let key_path = temp_path("wrong.key");
    std::fs::write(&key_path, format!("{}\n", "42".repeat(32))).unwrap();
    let out = std::process::Command::new(EXE)
        .args(["--tokens", "2", "--key-file", key_path.to_str().unwrap()])
        .env_remove("ESLM_KEY")
        .output()
        .unwrap();
    let _ = std::fs::remove_file(&key_path);
    assert_eq!(out.status.code(), Some(2));
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("authentication failed"), "unexpected error: {err}");
    assert!(!String::from_utf8_lossy(&out.stdout).contains("The ephemeral"), "generated text despite a wrong key");
}

#[test]
fn test_cli_gen_key_and_seal_round_trip() {
    let key_path = temp_path("fresh.key");
    let sealed_path = temp_path("fresh.enc");
    let _ = std::fs::remove_file(&key_path);
    let _ = std::fs::remove_file(&sealed_path);
    let run = |args: &[&str]| {
        std::process::Command::new(EXE).args(args).env_remove("ESLM_KEY").output().unwrap()
    };
    let (kp, sp) = (key_path.to_str().unwrap(), sealed_path.to_str().unwrap());
    let plain_path = concat!(env!("CARGO_MANIFEST_DIR"), "/assets/weights.bin");

    let out = run(&["--gen-key", kp]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    // A second --gen-key must refuse to overwrite the key.
    assert!(!run(&["--gen-key", kp]).status.success());

    let out = run(&["--seal", plain_path, sp, "--key-file", kp]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    // Refuses to overwrite without --force, allows it with.
    assert!(!run(&["--seal", plain_path, sp, "--key-file", kp]).status.success());
    assert!(run(&["--seal", plain_path, sp, "--key-file", kp, "--force"]).status.success());

    let sealed = std::fs::read(&sealed_path).unwrap();
    let key = SecretKey::from_file(kp).unwrap();
    assert_eq!(SealedPayload::parse(&sealed).unwrap().open_all(&key).unwrap(), PLAIN);
    // The demo key must not open a file sealed under a fresh key.
    assert!(SealedPayload::parse(&sealed).unwrap().open_all(&demo_key()).is_err());

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(&key_path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600, "generated key file must be private");
    }
    let _ = std::fs::remove_file(&key_path);
    let _ = std::fs::remove_file(&sealed_path);
}

/// End-to-end process-memory scan. Each scenario runs in its own child process; the
/// parent exits non-zero if a positive control finds nothing (scanner broken) or if a
/// scrubbed scenario still contains decompressed or decrypted weights.
#[test]
fn test_process_memory_forensics() {
    let out = std::process::Command::new(EXE)
        .args(["--forensics", "--key-file", KEY_PATH])
        .env_remove("ESLM_KEY")
        .output()
        .expect("failed to run the binary");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "forensics failed:\n{stdout}\n{}", String::from_utf8_lossy(&out.stderr));
    if forensics::is_supported() {
        assert!(stdout.contains("all assertions held"), "{stdout}");
    }
}
