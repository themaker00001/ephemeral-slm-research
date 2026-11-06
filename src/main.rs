use ephemeral_slm_research::{
    get_process_rss_bytes,
    EphemeralSLM, LifecycleEvent, MonolithicSLM, TelemetryLogger,
};
use std::env;
use std::io::{self, Write};
use std::time::Instant;

struct CliArgs {
    prompt: String,
    tokens: usize,
    mode: String,
    benchmark: bool,
    verify: bool,
    verbose: bool,
}

fn parse_args() -> CliArgs {
    let args: Vec<String> = env::args().collect();
    let mut prompt = "The ephemeral ".to_string();
    let mut tokens = 35;
    let mut mode = "ephemeral".to_string();
    let mut benchmark = false;
    let mut verify = false;
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
            "--benchmark" | "-b" => {
                benchmark = true;
            }
            "--verify" => {
                verify = true;
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
        verify,
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
    println!("        --verify           Run cryptographic/volatile zeroization audit");
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

fn run_side_by_side_benchmark(prompt: &str, num_tokens: usize) {
    println!("==================================================================");
    println!("         EPHEMERAL SLM RESEARCH - BENCHMARK & AUDIT");
    println!("==================================================================");
    println!("Prompt:  \"{}\"", prompt);
    println!("Tokens:  {}", num_tokens);
    println!();

    println!("[1/2] Executing Monolithic Baseline...");
    let mono_start = Instant::now();
    let mut mono = MonolithicSLM::new();
    let mono_alloc_time = mono_start.elapsed();

    let mut mono_tokens: Vec<u8> = prompt.bytes().filter(|&b| b < 128).collect();
    let mono_gen_start = Instant::now();
    for _ in 0..num_tokens {
        let ctx_start = mono_tokens.len().saturating_sub(mono.config.ctx_len);
        let ctx = &mono_tokens[ctx_start..];
        let logits = mono.forward(ctx);
        let best = logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
        mono_tokens.push(best as u8);
    }
    let mono_gen_time = mono_gen_start.elapsed();
    let mono_out = String::from_utf8_lossy(&mono_tokens).to_string();
    let mono_peak_bytes = mono.total_param_bytes;
    mono.cleanup();

    println!("[2/2] Executing Ephemeral Streaming Model...");
    let ephem_start = Instant::now();
    let mut ephem = EphemeralSLM::new();
    let telemetry = TelemetryLogger::new(false);
    let ephem_init_time = ephem_start.elapsed();

    let mut ephem_tokens: Vec<u8> = prompt.bytes().filter(|&b| b < 128).collect();
    let ephem_gen_start = Instant::now();
    for _ in 0..num_tokens {
        let ctx_start = ephem_tokens.len().saturating_sub(ephem.config().ctx_len);
        let ctx = &ephem_tokens[ctx_start..];
        let logits = ephem.forward_ephemeral(ctx, &telemetry);
        let best = logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
        ephem_tokens.push(best as u8);
    }
    let ephem_gen_time = ephem_gen_start.elapsed();
    let ephem_out = String::from_utf8_lossy(&ephem_tokens).to_string();

    println!();
    println!("========================== RESULTS ==============================");
    println!("Metric                     | Monolithic        | Ephemeral");
    println!("---------------------------+-------------------+------------------");
    println!("Active Param Residency     | {:>13} KB | {:>13} KB", mono_peak_bytes / 1024, ephem.metrics.peak_active_bytes / 1024);
    println!("Post-Run Memory Footprint  | {:>13} KB | {:>13} KB", mono_peak_bytes / 1024, ephem.metrics.current_active_bytes / 1024);
    println!("Total Bytes Wiped          | {:>13} B  | {:>13} B ", mono_peak_bytes, ephem.metrics.total_bytes_scrubbed);
    println!("Volatile Memory Audits     |           N/A     | {:>14} OK", ephem.metrics.audits_passed);
    println!("Init / Load Latency        | {:>14.2?} | {:>14.2?}", mono_alloc_time, ephem_init_time);
    println!("Generation Latency         | {:>14.2?} | {:>14.2?}", mono_gen_time, ephem_gen_time);
    println!("Per-Token Latency          | {:>11.2?} /t | {:>11.2?} /t", mono_gen_time / num_tokens as u32, ephem_gen_time / num_tokens as u32);
    println!("Output Text Parity         | {}", if mono_out == ephem_out { "100% IDENTICAL (EXACT MATCH)" } else { "MISMATCH" });
    println!("==================================================================");
    println!("Generated text:\n\"{}\"", ephem_out);
}

fn main() {
    let args = parse_args();

    if args.verify {
        println!("[*] Running Cryptographic & Memory Zeroization Verification...");
        let mut buf = ephemeral_slm_research::ephemeral::SecureBuffer::<f32>::zeroed(1024);
        for (i, v) in buf.iter_mut().enumerate() {
            *v = (i + 1) as f32;
        }
        assert!(!buf.audit_is_zero());
        buf.scrub();
        assert!(buf.audit_is_zero());
        println!("[✓] Secure volatile buffer scrubbing verified with compiler barrier.");
        return;
    }

    if args.benchmark {
        run_side_by_side_benchmark(&args.prompt, args.tokens);
        return;
    }

    match args.mode.as_str() {
        "monolithic" => run_monolithic_generation(&args.prompt, args.tokens, args.verbose),
        _ => run_ephemeral_generation(&args.prompt, args.tokens, args.verbose),
    }
}
