use ephemeral_slm_research::{
    get_process_rss_bytes,
    EphemeralSLM, LifecycleEvent, MonolithicSLM, TelemetryLogger,
};
use ephemeral_slm_research::forensics;
use std::env;
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
    forensics_child: Option<(String, String)>,
    verbose: bool,
}

fn parse_args() -> CliArgs {
    let args: Vec<String> = env::args().collect();
    let mut prompt = "The ephemeral ".to_string();
    let mut tokens = 35;
    let mut mode = "ephemeral".to_string();
    let mut benchmark = false;
    let mut reps = 15;
    let mut verify = false;
    let mut forensics = false;
    let mut forensics_child = None;
    let mut verbose = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--prompt" | "-p" if i + 1 < args.len() => {
                prompt = args[i + 1].clone();
                i += 1;
            }
            "--tokens" | "-n" if i + 1 < args.len() => {
                tokens = args[i + 1].parse().unwrap_or(35);
                i += 1;
            }
            "--mode" | "-m" if i + 1 < args.len() => {
                mode = args[i + 1].to_lowercase();
                i += 1;
            }
            "--reps" if i + 1 < args.len() => {
                reps = args[i + 1].parse().unwrap_or(15).max(1);
                i += 1;
            }
            "--benchmark" | "-b" => {
                benchmark = true;
            }
            "--verify" => {
                verify = true;
            }
            "--forensics" => {
                forensics = true;
            }
            // Internal: used by `--forensics` to run one scenario in a fresh process.
            "--forensics-child" if i + 2 < args.len() => {
                forensics_child = Some((args[i + 1].clone(), args[i + 2].clone()));
                i += 2;
            }
            "--verbose" | "-v" => {
                verbose = true;
            }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            _ => {}
        }
        i += 1;
    }

    CliArgs {
        prompt,
        tokens,
        mode,
        benchmark,
        reps,
        verify,
        forensics,
        forensics_child,
        verbose,
    }
}

fn print_help() {
    println!("Ephemeral SLM Research Demo (Concrete Implementation)");
    println!();
    println!("USAGE:");
    println!("    cargo run -- [OPTIONS]");
    println!();
    println!("OPTIONS:");
    println!("    -p, --prompt <TEXT>    Initial text prompt (default: 'The ephemeral ')");
    println!("    -n, --tokens <NUM>     Number of tokens to generate (default: 35)");
    println!("    -m, --mode <MODE>      Execution mode: 'ephemeral' (default) or 'monolithic'");
    println!("    -b, --benchmark        Run side-by-side benchmark comparing monolithic vs ephemeral");
    println!("        --reps <NUM>       Benchmark repetitions after warm-up (default: 15)");
    println!("        --verify           Check that SecureBuffer scrubbing zeroes a buffer");
    println!("        --forensics        Scan process memory for residual weights (macOS only)");
    println!("    -v, --verbose          Print micro-level per-layer lifecycle telemetry");
    println!("    -h, --help             Print help information");
}

fn run_ephemeral_generation(prompt: &str, num_tokens: usize, verbose: bool) {
    let telemetry = TelemetryLogger::new(verbose);
    let mut model = EphemeralSLM::new();
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
    println!(" Parameters: 50,928 weights (INT8 compressed)");
    println!(" Embedded:   {} bytes (.rodata)", model.payload_bytes());
    println!(" RSS Init:   {} KB", init_rss);
    println!("--------------------------------------------------");
    println!("Generating text from prompt: \"{}\"", prompt);
    print!("{}", prompt);
    io::stdout().flush().unwrap();

    let mut token_ids: Vec<u8> = prompt.bytes().filter(|&b| b < 128).collect();
    if token_ids.is_empty() {
        token_ids.push(b' ');
    }

    let gen_start = Instant::now();

    for step in 0..num_tokens {
        let step_start = Instant::now();
        let ctx_start = token_ids.len().saturating_sub(cfg.ctx_len);
        let ctx = &token_ids[ctx_start..];

        let logits = model.forward_ephemeral(ctx, &telemetry);

        // Greedy argmax selection
        let mut best_idx = 0;
        let mut max_logit = f32::NEG_INFINITY;
        for (idx, &val) in logits.iter().enumerate() {
            if val > max_logit {
                max_logit = val;
                best_idx = idx;
            }
        }

        let next_token = best_idx as u8;
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
    println!(" - Total bytes scrubbed:  {} bytes across {} operations", model.metrics.total_bytes_scrubbed, model.metrics.scrub_operations);
    println!(" - Forensic zero-audits:  {} passed, {} failed", model.metrics.audits_passed, model.metrics.audits_failed);
    println!(" - Final process RSS:     {} KB", final_rss);
}

fn run_monolithic_generation(prompt: &str, num_tokens: usize, verbose: bool) {
    let telemetry = TelemetryLogger::new(verbose);
    let init_rss = get_process_rss_bytes() / 1024;
    let mut model = MonolithicSLM::new();
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

    let mut token_ids: Vec<u8> = prompt.bytes().filter(|&b| b < 128).collect();
    if token_ids.is_empty() {
        token_ids.push(b' ');
    }

    let gen_start = Instant::now();

    for step in 0..num_tokens {
        let step_start = Instant::now();
        let ctx_start = token_ids.len().saturating_sub(model.config.ctx_len);
        let ctx = &token_ids[ctx_start..];

        let logits = model.forward(ctx);

        let mut best_idx = 0;
        let mut max_logit = f32::NEG_INFINITY;
        for (idx, &val) in logits.iter().enumerate() {
            if val > max_logit {
                max_logit = val;
                best_idx = idx;
            }
        }

        let next_token = best_idx as u8;
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
}

struct EphemRun {
    init: Duration,
    gen: Duration,
    out: Vec<u8>,
    metrics: ephemeral_slm_research::EphemeralMetrics,
}

fn bench_monolithic(prompt: &str, num_tokens: usize) -> MonoRun {
    let t = Instant::now();
    let mut model = MonolithicSLM::new();
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
    // Same volatile wipe the ephemeral engine uses, so only the residency differs.
    let t = Instant::now();
    model.cleanup();
    let wipe = t.elapsed();
    MonoRun { init, gen, wipe, out: ids, resident_bytes }
}

fn bench_ephemeral(prompt: &str, num_tokens: usize) -> EphemRun {
    let telemetry = TelemetryLogger::new(false);
    let t = Instant::now();
    let mut model = EphemeralSLM::new();
    let init = t.elapsed();

    let mut ids = prompt_ids(prompt);
    let ctx_len = model.config().ctx_len;
    let t = Instant::now();
    for _ in 0..num_tokens {
        let start = ids.len().saturating_sub(ctx_len);
        let logits = model.forward_ephemeral(&ids[start..], &telemetry);
        ids.push(argmax(&logits));
    }
    let gen = t.elapsed();
    EphemRun { init, gen, out: ids, metrics: model.metrics.clone() }
}

fn median(v: &mut [Duration]) -> Duration {
    v.sort();
    v[v.len() / 2]
}

fn run_side_by_side_benchmark(prompt: &str, num_tokens: usize, reps: usize) {
    const WARMUP: usize = 3;
    let num_tokens = num_tokens.max(1);

    println!("==================================================================");
    println!("         EPHEMERAL SLM RESEARCH - BENCHMARK");
    println!("==================================================================");
    println!("Prompt: \"{}\"   Tokens: {}   Warm-up: {}   Timed runs: {}", prompt, num_tokens, WARMUP, reps);
    println!("Order alternates every run; both engines use the same volatile wipe.");
    println!();

    for _ in 0..WARMUP {
        let _ = bench_monolithic(prompt, num_tokens);
        let _ = bench_ephemeral(prompt, num_tokens);
    }

    let mut m_init = Vec::new();
    let mut m_gen = Vec::new();
    let mut m_wipe = Vec::new();
    let mut e_init = Vec::new();
    let mut e_gen = Vec::new();
    let mut parity = true;
    let mut last_m: Option<MonoRun> = None;
    let mut last_e: Option<EphemRun> = None;

    for rep in 0..reps {
        let (m, e) = if rep % 2 == 0 {
            let m = bench_monolithic(prompt, num_tokens);
            let e = bench_ephemeral(prompt, num_tokens);
            (m, e)
        } else {
            let e = bench_ephemeral(prompt, num_tokens);
            let m = bench_monolithic(prompt, num_tokens);
            (m, e)
        };
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
    let kb = |b: usize| format!("{:.1} KB", b as f64 / 1024.0);
    let payload_kb = kb(ephemeral_slm_research::EMBEDDED_MODEL_PAYLOAD.len());
    let n = num_tokens as u32;

    println!("=========================== RESULTS ================================");
    println!("Metric                              | Monolithic         | Ephemeral");
    println!("------------------------------------+--------------------+-----------------");
    println!("Decompressed weights, peak resident | {:>18} | {:>15}", kb(m.resident_bytes), kb(e.metrics.peak_active_bytes));
    println!("Decompressed weights after run      | {:>18} | {:>15}", format!("{} (zeroed)", kb(m.resident_bytes)), kb(e.metrics.current_active_bytes));
    println!("INT8 payload in .rodata (always)    | {:>18} | {:>15}", payload_kb, payload_kb);
    println!("Bytes volatile-wiped per run        | {:>18} | {:>15}", m.resident_bytes, e.metrics.total_bytes_scrubbed);
    println!("Zero-audits passed / failed         | {:>18} | {:>15}", "n/a", format!("{} / {}", e.metrics.audits_passed, e.metrics.audits_failed));
    println!("Upfront cost (median)               | {:>18.2?} | {:>15.2?}", m_init_med, e_init_med);
    println!("Wipe cost, explicit (median)        | {:>18.2?} | {:>15}", m_wipe_med, "in generation");
    println!("Generation (median)                 | {:>18.2?} | {:>15.2?}", m_gen_med, e_gen_med);
    println!("Generation (min .. max)             | {:>18} | {:>15}", format!("{:.1?}..{:.1?}", m_gen_min, m_gen_max), format!("{:.1?}..{:.1?}", e_gen_min, e_gen_max));
    println!("Per token (median)                  | {:>18.2?} | {:>15.2?}", m_gen_med / n, e_gen_med / n);
    println!("Init + generation + wipe (median)   | {:>18.2?} | {:>15.2?}", m_init_med + m_gen_med + m_wipe_med, e_init_med + e_gen_med);
    println!("Output parity (all runs)            | {}", if parity { "IDENTICAL" } else { "MISMATCH" });
    println!("====================================================================");
    println!("Notes:");
    println!(" - The INT8 payload stays resident in both engines; only the decompressed");
    println!("   f32 working copies differ. This is not model confidentiality.");
    println!(" - If the min..max ranges overlap, the latency difference is not significant.");
    println!(" - Run with --release; debug builds are not representative.");
    println!("Generated text:\n\"{}\"", String::from_utf8_lossy(&e.out));
}

fn main() {
    let args = parse_args();

    if let Some((scenario, sig_hex)) = &args.forensics_child {
        std::process::exit(forensics::run_child(scenario, sig_hex));
    }

    if args.forensics {
        std::process::exit(forensics::run_parent());
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
        println!("     process, run: cargo run --release -- --forensics");
        return;
    }

    if args.benchmark {
        run_side_by_side_benchmark(&args.prompt, args.tokens, args.reps);
        return;
    }

    match args.mode.as_str() {
        "monolithic" => run_monolithic_generation(&args.prompt, args.tokens, args.verbose),
        _ => run_ephemeral_generation(&args.prompt, args.tokens, args.verbose),
    }
}
