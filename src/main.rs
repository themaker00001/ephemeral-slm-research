use ephemeral_slm_research::forensics;
use ephemeral_slm_research::seal::{self, SealedPayload, SecretKey};
use ephemeral_slm_research::{
    get_process_rss_bytes, EphemeralSLM, LifecycleEvent, MonolithicSLM, SecureBuffer, TelemetryLogger,
    EMBEDDED_SEALED_PAYLOAD,
};
use std::env;
use std::fmt::Display;
use std::io::{self, Write};
use std::time::{Duration, Instant};

struct CliArgs {
    prompt: String,
    tokens: usize,
    mode: String,
    benchmark: bool,
    reps: usize,
    verify: bool,
    forensics: bool,
    forensics_child: Option<(String, String, String)>,
    verbose: bool,
    key_file: Option<String>,
    seal: Option<(String, String)>,
    force: bool,
    gen_key: Option<String>,
}

fn parse_args() -> CliArgs {
    let args: Vec<String> = env::args().collect();
    let mut cli = CliArgs {
        prompt: "The ephemeral ".to_string(),
        tokens: 35,
        mode: "ephemeral".to_string(),
        benchmark: false,
        reps: 15,
        verify: false,
        forensics: false,
        forensics_child: None,
        verbose: false,
        key_file: None,
        seal: None,
        force: false,
        gen_key: None,
    };

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--prompt" | "-p" if i + 1 < args.len() => {
                cli.prompt = args[i + 1].clone();
                i += 1;
            }
            "--tokens" | "-n" if i + 1 < args.len() => {
                cli.tokens = args[i + 1].parse().unwrap_or(35);
                i += 1;
            }
            "--mode" | "-m" if i + 1 < args.len() => {
                cli.mode = args[i + 1].to_lowercase();
                i += 1;
            }
            "--reps" if i + 1 < args.len() => {
                cli.reps = args[i + 1].parse().unwrap_or(15).max(1);
                i += 1;
            }
            "--key-file" | "-k" if i + 1 < args.len() => {
                cli.key_file = Some(args[i + 1].clone());
                i += 1;
            }
            "--seal" if i + 2 < args.len() => {
                cli.seal = Some((args[i + 1].clone(), args[i + 2].clone()));
                i += 2;
            }
            "--gen-key" if i + 1 < args.len() => {
                cli.gen_key = Some(args[i + 1].clone());
                i += 1;
            }
            "--force" => {
                cli.force = true;
            }
            "--benchmark" | "-b" => {
                cli.benchmark = true;
            }
            "--verify" => {
                cli.verify = true;
            }
            "--forensics" => {
                cli.forensics = true;
            }
            // Internal: used by `--forensics` to run one scenario in a fresh process.
            "--forensics-child" if i + 3 < args.len() => {
                cli.forensics_child =
                    Some((args[i + 1].clone(), args[i + 2].clone(), args[i + 3].clone()));
                i += 3;
            }
            "--verbose" | "-v" => {
                cli.verbose = true;
            }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            _ => {}
        }
        i += 1;
    }
    cli
}

fn print_help() {
    println!("Ephemeral SLM Research (sealed model, layer-by-layer decryption)");
    println!();
    println!("USAGE:");
    println!("    cargo run --release -- --key-file assets/demo.key [OPTIONS]");
    println!();
    println!("The model is embedded encrypted. Every mode that runs it needs the key:");
    println!("    -k, --key-file <PATH>  File holding the key as 64 hex characters");
    println!("                           (or set ESLM_KEY; the demo key is assets/demo.key)");
    println!();
    println!("OPTIONS:");
    println!("    -p, --prompt <TEXT>    Initial text prompt (default: 'The ephemeral ')");
    println!("    -n, --tokens <NUM>     Number of tokens to generate (default: 35)");
    println!("    -m, --mode <MODE>      Execution mode: 'ephemeral' (default) or 'monolithic'");
    println!("    -b, --benchmark        Run side-by-side benchmark comparing monolithic vs ephemeral");
    println!("        --reps <NUM>       Benchmark repetitions after warm-up (default: 15)");
    println!("        --forensics        Scan process memory for residual weights (macOS only)");
    println!("        --verify           Check that SecureBuffer scrubbing zeroes a buffer (no key needed)");
    println!("    -v, --verbose          Print micro-level per-layer lifecycle telemetry");
    println!();
    println!("KEY AND SEALING TOOLS (no model is run):");
    println!("        --gen-key <PATH>   Write a new random key (refuses to overwrite)");
    println!("        --seal <IN> <OUT>  Encrypt a plaintext weights.bin into a sealed file");
    println!("                           (add --force to overwrite OUT). Rebuild afterwards.");
    println!();
    println!("    -h, --help             Print help information");
}

fn fail(msg: impl Display) -> ! {
    eprintln!("error: {msg}");
    std::process::exit(2);
}

fn load_key(path: Option<&str>) -> SecretKey {
    SecretKey::from_cli(path).unwrap_or_else(|e| fail(e))
}

fn run_ephemeral_generation(prompt: &str, num_tokens: usize, verbose: bool, key: &SecretKey) {
    let telemetry = TelemetryLogger::new(verbose);
    let mut model = EphemeralSLM::open(key).unwrap_or_else(|e| fail(e));
    let cfg = model.config();
    let init_rss = get_process_rss_bytes() / 1024;

    telemetry.log(LifecycleEvent::ModelInitialized {
        mode: "ephemeral",
        total_parameters: 50928,
        payload_bytes: model.payload_bytes(),
        initial_rss_kb: init_rss,
    });

    println!("--------------------------------------------------");
    println!(" Mode:       EPHEMERAL (Transient Streaming)");
    println!(" Model:      2-Layer LLaMA-style MicroTransformer");
    println!(" Parameters: 50,928 weights (INT8, encrypted at rest)");
    println!(" Embedded:   {} bytes sealed (.rodata, ChaCha20-Poly1305)", model.payload_bytes());
    println!(" RSS Init:   {} KB", init_rss);
    println!("--------------------------------------------------");
    println!("Generating text from prompt: \"{}\"", prompt);
    print!("{}", prompt);
    io::stdout().flush().unwrap();

    let mut token_ids = prompt_ids(prompt);

    let gen_start = Instant::now();

    for step in 0..num_tokens {
        let step_start = Instant::now();
        let ctx_start = token_ids.len().saturating_sub(cfg.ctx_len);
        let ctx = &token_ids[ctx_start..];

        let logits = model
            .forward_ephemeral(ctx, &telemetry)
            .unwrap_or_else(|e| fail(e));

        let next_token = argmax(&logits);
        token_ids.push(next_token);

        let ch = next_token as char;
        print!("{}", ch);
        io::stdout().flush().unwrap();

        telemetry.log(LifecycleEvent::TokenStep {
            step,
            token: next_token,
            token_char: ch,
            elapsed_us: step_start.elapsed().as_micros(),
            active_memory_kb: model.metrics.current_active_bytes / 1024,
        });
    }

    println!("\n--------------------------------------------------");
    let total_us = gen_start.elapsed().as_micros();
    let final_rss = get_process_rss_bytes() / 1024;

    telemetry.log(LifecycleEvent::CleanupCompleted {
        mode: "ephemeral",
        total_scrubbed_bytes: model.metrics.total_bytes_scrubbed,
        audits_passed: model.metrics.audits_passed,
        final_rss_kb: final_rss,
    });

    println!("Summary:");
    println!(" - Tokens generated:      {}", num_tokens);
    println!(" - Total time:            {:.2} ms ({:.2} ms/token)", total_us as f64 / 1000.0, total_us as f64 / 1000.0 / num_tokens as f64);
    println!(" - Peak parameter in RAM: {} KB (only 1 layer at any instant!)", model.metrics.peak_active_bytes / 1024);
    println!(" - Active params now:     {} KB (ephemeral eradication)", model.metrics.current_active_bytes / 1024);
    println!(" - Sections decrypted:    {} ({} KB of INT8 plaintext, each wiped right after use)", model.metrics.sections_opened, model.metrics.plaintext_bytes_wiped / 1024);
    println!(" - Total bytes scrubbed:  {} bytes across {} operations", model.metrics.total_bytes_scrubbed, model.metrics.scrub_operations);
    println!(" - Forensic zero-audits:  {} passed, {} failed", model.metrics.audits_passed, model.metrics.audits_failed);
    println!(" - Final process RSS:     {} KB", final_rss);
}

fn run_monolithic_generation(prompt: &str, num_tokens: usize, verbose: bool, key: &SecretKey) {
    let telemetry = TelemetryLogger::new(verbose);
    let init_rss = get_process_rss_bytes() / 1024;
    let mut model = MonolithicSLM::open(key).unwrap_or_else(|e| fail(e));
    let post_alloc_rss = get_process_rss_bytes() / 1024;

    telemetry.log(LifecycleEvent::ModelInitialized {
        mode: "monolithic",
        total_parameters: 50928,
        payload_bytes: model.total_param_bytes,
        initial_rss_kb: post_alloc_rss,
    });

    println!("--------------------------------------------------");
    println!(" Mode:       MONOLITHIC (Full RAM Residency)");
    println!(" Model:      2-Layer LLaMA-style MicroTransformer");
    println!(" Parameters: 50,928 weights (all resident in RAM)");
    println!(" RAM Allocated: {} KB permanent heap", model.total_param_bytes / 1024);
    println!(" RSS Init:   {} KB (pre-alloc: {} KB)", post_alloc_rss, init_rss);
    println!("--------------------------------------------------");
    println!("Generating text from prompt: \"{}\"", prompt);
    print!("{}", prompt);
    io::stdout().flush().unwrap();

    let mut token_ids = prompt_ids(prompt);

    let gen_start = Instant::now();

    for step in 0..num_tokens {
        let step_start = Instant::now();
        let ctx_start = token_ids.len().saturating_sub(model.config.ctx_len);
        let ctx = &token_ids[ctx_start..];

        let logits = model.forward(ctx);

        let next_token = argmax(&logits);
        token_ids.push(next_token);

        let ch = next_token as char;
        print!("{}", ch);
        io::stdout().flush().unwrap();

        telemetry.log(LifecycleEvent::TokenStep {
            step,
            token: next_token,
            token_char: ch,
            elapsed_us: step_start.elapsed().as_micros(),
            active_memory_kb: model.total_param_bytes / 1024,
        });
    }

    println!("\n--------------------------------------------------");
    let total_us = gen_start.elapsed().as_micros();
    model.cleanup();
    let final_rss = get_process_rss_bytes() / 1024;

    telemetry.log(LifecycleEvent::CleanupCompleted {
        mode: "monolithic",
        total_scrubbed_bytes: model.total_param_bytes,
        audits_passed: 1,
        final_rss_kb: final_rss,
    });

    println!("Summary:");
    println!(" - Tokens generated:      {}", num_tokens);
    println!(" - Total time:            {:.2} ms ({:.2} ms/token)", total_us as f64 / 1000.0, total_us as f64 / 1000.0 / num_tokens as f64);
    println!(" - Retained in RAM:       {} KB (throughout execution)", model.total_param_bytes / 1024);
    println!(" - INT8 plaintext wiped:  {} KB (at load, after dequantization)", model.plaintext_bytes_wiped / 1024);
    println!(" - Final process RSS:     {} KB", final_rss);
}

fn prompt_ids(prompt: &str) -> Vec<u8> {
    let mut ids: Vec<u8> = prompt.bytes().filter(|&b| b < 128).collect();
    if ids.is_empty() {
        ids.push(b' ');
    }
    ids
}

fn argmax(logits: &[f32]) -> u8 {
    let mut best = 0;
    for (i, &v) in logits.iter().enumerate() {
        if v > logits[best] {
            best = i;
        }
    }
    best as u8
}

struct MonoRun {
    init: Duration,
    gen: Duration,
    wipe: Duration,
    out: Vec<u8>,
    resident_bytes: usize,
    plaintext_wiped: usize,
}

struct EphemRun {
    init: Duration,
    gen: Duration,
    out: Vec<u8>,
    metrics: ephemeral_slm_research::EphemeralMetrics,
}

fn bench_monolithic(prompt: &str, num_tokens: usize, key: &SecretKey) -> MonoRun {
    // Construction includes decrypting every section once.
    let t = Instant::now();
    let mut model = MonolithicSLM::open(key).unwrap_or_else(|e| fail(e));
    let init = t.elapsed();

    let mut ids = prompt_ids(prompt);
    let t = Instant::now();
    for _ in 0..num_tokens {
        let start = ids.len().saturating_sub(model.config.ctx_len);
        let logits = model.forward(&ids[start..]);
        ids.push(argmax(&logits));
    }
    let gen = t.elapsed();

    let resident_bytes = model.total_param_bytes;
    let plaintext_wiped = model.plaintext_bytes_wiped;
    // Same volatile wipe the ephemeral engine uses, so only the residency differs.
    let t = Instant::now();
    model.cleanup();
    let wipe = t.elapsed();
    MonoRun { init, gen, wipe, out: ids, resident_bytes, plaintext_wiped }
}

fn bench_ephemeral(prompt: &str, num_tokens: usize, key: &SecretKey) -> EphemRun {
    let telemetry = TelemetryLogger::new(false);
    let t = Instant::now();
    let mut model = EphemeralSLM::open(key).unwrap_or_else(|e| fail(e));
    let init = t.elapsed();

    let mut ids = prompt_ids(prompt);
    let ctx_len = model.config().ctx_len;
    let t = Instant::now();
    for _ in 0..num_tokens {
        let start = ids.len().saturating_sub(ctx_len);
        let logits = model
            .forward_ephemeral(&ids[start..], &telemetry)
            .unwrap_or_else(|e| fail(e));
        ids.push(argmax(&logits));
    }
    let gen = t.elapsed();
    EphemRun { init, gen, out: ids, metrics: model.metrics.clone() }
}

/// Time to authenticate and decrypt every section once (what the ephemeral engine repeats
/// on every token), measured in isolation.
fn bench_decrypt_all(key: &SecretKey) -> Duration {
    let sealed = SealedPayload::parse(EMBEDDED_SEALED_PAYLOAD).unwrap_or_else(|e| fail(e));
    let mut bufs: Vec<SecureBuffer<u8>> = (0..sealed.section_count())
        .map(|i| SecureBuffer::zeroed(sealed.section_plain_len(i)))
        .collect();
    let t = Instant::now();
    for (i, b) in bufs.iter_mut().enumerate() {
        sealed.open_section(i, key, b).unwrap_or_else(|e| fail(e));
    }
    let d = t.elapsed();
    for b in bufs.iter_mut() {
        b.scrub();
    }
    d
}

fn median(v: &mut [Duration]) -> Duration {
    v.sort();
    v[v.len() / 2]
}

fn run_side_by_side_benchmark(prompt: &str, num_tokens: usize, reps: usize, key: &SecretKey) {
    const WARMUP: usize = 3;
    let num_tokens = num_tokens.max(1);

    println!("==================================================================");
    println!("         EPHEMERAL SLM RESEARCH - BENCHMARK");
    println!("==================================================================");
    println!("Prompt: \"{}\"   Tokens: {}   Warm-up: {}   Timed runs: {}", prompt, num_tokens, WARMUP, reps);
    println!("Order alternates every run; both engines use the same volatile wipe.");
    println!("Both decrypt from the sealed payload: monolithic once, ephemeral per token.");
    println!();

    for _ in 0..WARMUP {
        let _ = bench_monolithic(prompt, num_tokens, key);
        let _ = bench_ephemeral(prompt, num_tokens, key);
        let _ = bench_decrypt_all(key);
    }

    let mut m_init = Vec::new();
    let mut m_gen = Vec::new();
    let mut m_wipe = Vec::new();
    let mut e_init = Vec::new();
    let mut e_gen = Vec::new();
    let mut dec = Vec::new();
    let mut parity = true;
    let mut last_m: Option<MonoRun> = None;
    let mut last_e: Option<EphemRun> = None;

    for rep in 0..reps {
        let (m, e) = if rep % 2 == 0 {
            let m = bench_monolithic(prompt, num_tokens, key);
            let e = bench_ephemeral(prompt, num_tokens, key);
            (m, e)
        } else {
            let e = bench_ephemeral(prompt, num_tokens, key);
            let m = bench_monolithic(prompt, num_tokens, key);
            (m, e)
        };
        dec.push(bench_decrypt_all(key));
        parity &= m.out == e.out;
        m_init.push(m.init);
        m_gen.push(m.gen);
        m_wipe.push(m.wipe);
        e_init.push(e.init);
        e_gen.push(e.gen);
        last_m = Some(m);
        last_e = Some(e);
    }
    let m = last_m.unwrap();
    let e = last_e.unwrap();

    let (m_gen_min, m_gen_max) = (*m_gen.iter().min().unwrap(), *m_gen.iter().max().unwrap());
    let (e_gen_min, e_gen_max) = (*e_gen.iter().min().unwrap(), *e_gen.iter().max().unwrap());
    let m_init_med = median(&mut m_init);
    let m_gen_med = median(&mut m_gen);
    let m_wipe_med = median(&mut m_wipe);
    let e_init_med = median(&mut e_init);
    let e_gen_med = median(&mut e_gen);
    let dec_med = median(&mut dec);
    let kb = |b: usize| format!("{:.1} KB", b as f64 / 1024.0);
    let payload_kb = kb(EMBEDDED_SEALED_PAYLOAD.len());
    let max_section = SealedPayload::parse(EMBEDDED_SEALED_PAYLOAD)
        .map(|s| s.max_section_len())
        .unwrap_or(0);
    let n = num_tokens as u32;

    println!("=========================== RESULTS ================================");
    println!("Metric                              | Monolithic         | Ephemeral");
    println!("------------------------------------+--------------------+-----------------");
    println!("Decompressed weights, peak resident | {:>18} | {:>15}", kb(m.resident_bytes), kb(e.metrics.peak_active_bytes));
    println!("Decompressed weights after run      | {:>18} | {:>15}", format!("{} (zeroed)", kb(m.resident_bytes)), kb(e.metrics.current_active_bytes));
    println!("Sealed payload in .rodata (always)  | {:>18} | {:>15}", payload_kb, payload_kb);
    println!("Decrypted INT8, most held at once   | {:>18} | {:>15}", kb(max_section), kb(e.metrics.peak_plaintext_bytes));
    println!("Decrypted INT8 wiped per run        | {:>18} | {:>15}", m.plaintext_wiped, e.metrics.plaintext_bytes_wiped);
    println!("Decompressed bytes wiped per run    | {:>18} | {:>15}", m.resident_bytes, e.metrics.total_bytes_scrubbed);
    println!("Zero-audits passed / failed         | {:>18} | {:>15}", "n/a", format!("{} / {}", e.metrics.audits_passed, e.metrics.audits_failed));
    println!("Upfront cost incl. decrypt (median) | {:>18.2?} | {:>15.2?}", m_init_med, e_init_med);
    println!("Wipe cost, explicit (median)        | {:>18.2?} | {:>15}", m_wipe_med, "in generation");
    println!("Generation (median)                 | {:>18.2?} | {:>15.2?}", m_gen_med, e_gen_med);
    println!("Generation (min .. max)             | {:>18} | {:>15}", format!("{:.1?}..{:.1?}", m_gen_min, m_gen_max), format!("{:.1?}..{:.1?}", e_gen_min, e_gen_max));
    println!("Per token (median)                  | {:>18.2?} | {:>15.2?}", m_gen_med / n, e_gen_med / n);
    println!("Init + generation + wipe (median)   | {:>18.2?} | {:>15.2?}", m_init_med + m_gen_med + m_wipe_med, e_init_med + e_gen_med);
    println!("Output parity (all runs)            | {}", if parity { "IDENTICAL" } else { "MISMATCH" });
    println!("====================================================================");
    println!("Notes:");
    println!(" - One full decrypt + authenticate of every section takes {:.1?} (median).", dec_med);
    println!("   Monolithic pays that once; ephemeral pays it on every token ({} here).", num_tokens);
    println!(" - The executable holds only ciphertext. The key is not embedded, but it stays");
    println!("   in process memory for the whole run in both engines.");
    println!(" - If the min..max ranges overlap, the latency difference is not significant.");
    println!(" - Run with --release; debug builds are not representative.");
    println!("Generated text:\n\"{}\"", String::from_utf8_lossy(&e.out));
}

fn run_gen_key(path: &str) {
    let key = SecretKey::random().unwrap_or_else(|e| fail(e));
    key.write_to_file(path).unwrap_or_else(|e| fail(e));
    println!("[ok] wrote a new 256-bit key to {path} (64 hex characters).");
    println!("     Keep it out of version control and out of the executable.");
}

fn run_seal(input: &str, output: &str, force: bool, key: &SecretKey) {
    if !force && std::path::Path::new(output).exists() {
        fail(format!("{output} already exists (use --force to overwrite)"));
    }
    let plain = std::fs::read(input).unwrap_or_else(|e| fail(format!("{input}: {e}")));
    let sealed = seal::seal_random_nonce(&plain, key).unwrap_or_else(|e| fail(e));
    // Read it back before writing anything: the sealed bytes must open to the same payload.
    let back = SealedPayload::parse(&sealed)
        .and_then(|s| s.open_all(key))
        .unwrap_or_else(|e| fail(format!("self-check failed: {e}")));
    if back != plain {
        fail("self-check failed: the sealed payload does not open to the input");
    }
    std::fs::write(output, &sealed).unwrap_or_else(|e| fail(format!("{output}: {e}")));
    println!(
        "[ok] sealed {} plaintext bytes into {} bytes at {output} (self-check passed).",
        plain.len(),
        sealed.len()
    );
    println!("     Rebuild so the new file is embedded: cargo build --release");
}

fn main() {
    let args = parse_args();

    if let Some((scenario, sig_hex, n_f32)) = &args.forensics_child {
        std::process::exit(forensics::run_child(scenario, sig_hex, n_f32, args.key_file.as_deref()));
    }

    if args.forensics {
        std::process::exit(forensics::run_parent(args.key_file.as_deref()));
    }

    if let Some(path) = &args.gen_key {
        run_gen_key(path);
        return;
    }

    if args.verify {
        println!("[*] Checking SecureBuffer volatile scrubbing...");
        let mut buf = ephemeral_slm_research::ephemeral::SecureBuffer::<f32>::zeroed(1024);
        for (i, v) in buf.iter_mut().enumerate() {
            *v = (i + 1) as f32;
        }
        assert!(!buf.audit_is_zero());
        buf.scrub();
        assert!(buf.audit_is_zero());
        println!("[ok] The scrubbed buffer reads back as all zero.");
        println!("     This checks the buffer itself only. For copies elsewhere in the");
        println!("     process, run: cargo run --release -- --forensics --key-file assets/demo.key");
        return;
    }

    if let Some((input, output)) = &args.seal {
        let key = load_key(args.key_file.as_deref());
        run_seal(input, output, args.force, &key);
        return;
    }

    let key = load_key(args.key_file.as_deref());

    if args.benchmark {
        run_side_by_side_benchmark(&args.prompt, args.tokens, args.reps, &key);
        return;
    }

    match args.mode.as_str() {
        "monolithic" => run_monolithic_generation(&args.prompt, args.tokens, args.verbose, &key),
        _ => run_ephemeral_generation(&args.prompt, args.tokens, args.verbose, &key),
    }
}
