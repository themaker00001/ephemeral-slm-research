//! Process-memory forensics: look for residual copies of decrypted or decompressed weights.
//!
//! The `SecureBuffer` audits only re-read the buffer that was just overwritten, so they
//! cannot tell whether a copy survives somewhere else. This module answers that question
//! empirically instead. It takes short byte signatures of two things the model passes
//! through on its way from the sealed payload to a forward pass:
//!
//! - **f32 signatures**: the *decompressed* weights (what the model looks like once rebuilt
//!   from INT8);
//! - **int8 signatures**: the *decrypted* INT8 weights (what a section looks like right
//!   after it comes out of the cipher).
//!
//! It then scans every readable and writable region of the current process's address space
//! for them, at every byte offset.
//!
//! How the experiment is kept honest:
//! - Each scenario runs in its **own child process**, so residue from one scenario cannot
//!   leak into another.
//! - Signatures are computed by the **parent** (which holds the key and decrypts the model
//!   once) and handed to the child as ASCII hex on the command line, so the child never has
//!   the raw bytes in memory except in one buffer whose address range is excluded from the
//!   scan. (ASCII hex cannot match.)
//! - Two **positive controls** deliberately leave weights in memory, so the scan must find
//!   them: `monolithic-live` for f32 and `plaintext-live` for decrypted INT8. If either
//!   finds nothing, the scanner is broken and the other results mean nothing.
//!
//! Scope: macOS only (Mach `mach_vm_region`). Read-only mappings are skipped: the sealed
//! payload in `.rodata` is ciphertext and could not match a plaintext signature anyway.
//! This does not look at registers, swap, hibernation images or other processes.

use crate::ephemeral::volatile_zero;
use crate::model::{EphemeralSLM, MonolithicSLM};
use crate::seal::{SealError, SealedPayload, SecretKey};
use crate::telemetry::TelemetryLogger;
use crate::weights::{
    parse_embedding_section, parse_final_section, parse_layer_section, PayloadView, QuantizedSlice,
    EMBEDDED_SEALED_PAYLOAD,
};

/// Signature length in bytes: 8 consecutive `f32` values, or 32 consecutive INT8 values.
pub const SIG_LEN: usize = 32;
const SIG_FLOATS: usize = SIG_LEN / 4;

/// Tokens generated in each scenario before the memory scan.
const SCENARIO_TOKENS: usize = 20;
const PROMPT: &[u8] = b"The ephemeral ";

/// What the parent expects of one class of signature in a scenario.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Expect {
    /// Must find residue; proves the scanner works.
    PositiveControl,
    /// Must find nothing.
    Clean,
    /// Reported but not asserted (depends on compiler and allocator behaviour).
    Informational,
}

impl Expect {
    fn holds(self, hits: usize) -> bool {
        match self {
            Expect::PositiveControl => hits > 0,
            Expect::Clean => hits == 0,
            Expect::Informational => true,
        }
    }

    fn tag(self) -> &'static str {
        match self {
            Expect::PositiveControl => ">0",
            Expect::Clean => "=0",
            Expect::Informational => "info",
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Scenario {
    pub name: &'static str,
    /// Expectation for decompressed f32 weights.
    pub f32_expect: Expect,
    /// Expectation for decrypted INT8 plaintext.
    pub i8_expect: Expect,
    pub desc: &'static str,
}

pub const SCENARIOS: &[Scenario] = &[
    Scenario {
        name: "monolithic-live",
        f32_expect: Expect::PositiveControl,
        i8_expect: Expect::Clean,
        desc: "all f32 weights resident; decrypted INT8 wiped at load",
    },
    Scenario {
        name: "plaintext-live",
        f32_expect: Expect::Clean,
        i8_expect: Expect::PositiveControl,
        desc: "every section decrypted into plain Vecs and kept",
    },
    Scenario {
        name: "streaming-no-scrub",
        f32_expect: Expect::Informational,
        i8_expect: Expect::Informational,
        desc: "decrypt + dequantize through plain Vecs, never zeroed",
    },
    Scenario {
        name: "monolithic-naive",
        f32_expect: Expect::Informational,
        i8_expect: Expect::Clean,
        desc: "fill(0.0) cleanup, model still allocated",
    },
    Scenario {
        name: "monolithic-naive-dropped",
        f32_expect: Expect::Informational,
        i8_expect: Expect::Clean,
        desc: "fill(0.0) cleanup, then drop the model",
    },
    Scenario {
        name: "monolithic-secure",
        f32_expect: Expect::Clean,
        i8_expect: Expect::Clean,
        desc: "volatile cleanup, model still allocated",
    },
    Scenario {
        name: "ephemeral",
        f32_expect: Expect::Clean,
        i8_expect: Expect::Clean,
        desc: "layer decrypt + volatile scrub, after generation",
    },
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SigKind {
    /// Decompressed f32 weights.
    F32,
    /// Decrypted INT8 plaintext.
    Int8,
}

/// A labelled byte signature.
#[derive(Debug, Clone)]
pub struct Signature {
    pub label: String,
    pub bytes: [u8; SIG_LEN],
    pub kind: SigKind,
}

/// Result of one memory scan.
#[derive(Debug, Clone, Default)]
pub struct ScanReport {
    /// Hits per signature, in the order the signatures were supplied.
    pub hits: Vec<usize>,
    pub regions_scanned: usize,
    pub bytes_scanned: u64,
}

impl ScanReport {
    pub fn total_hits(&self) -> usize {
        self.hits.iter().sum()
    }
}

// ---------------------------------------------------------------------------
// Signatures and hex transport
// ---------------------------------------------------------------------------

fn signature_for_f32(label: &str, tensor: &QuantizedSlice) -> Option<Signature> {
    let mut vals = vec![0.0_f32; tensor.num_elements()];
    tensor.dequantize_into(&mut vals);
    if vals.len() < SIG_FLOATS {
        return None;
    }
    // Start in the middle of the tensor: allocators overwrite the first bytes of a freed
    // block with free-list pointers, which would silently destroy a signature taken from
    // the start and hide residue in freed memory.
    let first = vals.len() / 2;
    for start in first..=vals.len() - SIG_FLOATS {
        let window = &vals[start..start + SIG_FLOATS];
        let mut distinct: Vec<u32> = window
            .iter()
            .filter(|v| **v != 0.0)
            .map(|v| v.to_bits())
            .collect();
        distinct.sort_unstable();
        distinct.dedup();
        // Require several distinct non-zero values so the signature is not a zero run
        // or a repeated constant that could match unrelated memory.
        if distinct.len() >= 6 {
            let mut bytes = [0u8; SIG_LEN];
            for (i, v) in window.iter().enumerate() {
                bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_ne_bytes());
            }
            return Some(Signature {
                label: label.to_string(),
                bytes,
                kind: SigKind::F32,
            });
        }
    }
    None
}

fn signature_for_i8(label: &str, tensor: &QuantizedSlice) -> Option<Signature> {
    let data = tensor.data;
    if data.len() < SIG_LEN {
        return None;
    }
    // Middle of the tensor, for the same allocator reason as the f32 signatures.
    let first = (data.len() / 2) & !3;
    for start in (first..=data.len() - SIG_LEN).step_by(4) {
        let window = &data[start..start + SIG_LEN];
        let mut seen = [false; 256];
        for &b in window {
            seen[b as u8 as usize] = true;
        }
        // Many distinct byte values, so the window cannot be a zero run or a short repeat.
        if seen.iter().filter(|&&s| s).count() >= 12 {
            let mut bytes = [0u8; SIG_LEN];
            for (i, &b) in window.iter().enumerate() {
                bytes[i] = b as u8;
            }
            return Some(Signature {
                label: format!("i8:{label}"),
                bytes,
                kind: SigKind::Int8,
            });
        }
    }
    None
}

/// Build signatures by decrypting the embedded payload. Call this in the parent only: it
/// holds the whole model in the clear for the duration of the call, then wipes it.
///
/// The f32 signatures come first, then the INT8 ones.
pub fn compute_signatures(key: &SecretKey) -> Result<Vec<Signature>, SealError> {
    let sealed = SealedPayload::parse(EMBEDDED_SEALED_PAYLOAD)?;
    let mut plain = sealed.open_all(key)?;
    let sigs = {
        let payload = PayloadView::parse(&plain)
            .map_err(|_| SealError::Malformed("decrypted payload does not parse"))?;
        let last = payload.config.n_layers - 1;
        let (tok, pos) = payload.get_embeddings();
        let l0 = payload.get_layer_slices(0);
        let ln = payload.get_layer_slices(last);
        let (_, head) = payload.get_final_slices();

        let tensors: [(&str, QuantizedSlice); 7] = [
            ("tok_emb", tok),
            ("pos_emb", pos),
            ("l0.wq", l0.wq),
            ("l0.w1", l0.w1),
            ("ln.wo", ln.wo),
            ("ln.w2", ln.w2),
            ("head", head),
        ];
        let mut sigs: Vec<Signature> = tensors
            .iter()
            .filter_map(|(label, t)| signature_for_f32(label, t))
            .collect();
        sigs.extend(tensors.iter().filter_map(|(label, t)| signature_for_i8(label, t)));
        sigs
    };
    volatile_zero(&mut plain);
    Ok(sigs)
}

pub fn to_hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        out.push(DIGITS[(b >> 4) as usize] as char);
        out.push(DIGITS[(b & 0xf) as usize] as char);
    }
    out
}

/// Decode hex into a buffer with exactly the right capacity (no reallocation, so there
/// is exactly one copy of the bytes in memory).
pub fn from_hex(hex: &str) -> Result<Vec<u8>, String> {
    if !hex.len().is_multiple_of(2) {
        return Err("odd-length hex string".into());
    }
    let nibble = |c: u8| -> Result<u8, String> {
        match c {
            b'0'..=b'9' => Ok(c - b'0'),
            b'a'..=b'f' => Ok(c - b'a' + 10),
            b'A'..=b'F' => Ok(c - b'A' + 10),
            _ => Err(format!("invalid hex digit {:?}", c as char)),
        }
    };
    let raw = hex.as_bytes();
    let mut out = Vec::with_capacity(raw.len() / 2);
    for pair in raw.chunks_exact(2) {
        out.push((nibble(pair[0])? << 4) | nibble(pair[1])?);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Memory scanner (macOS)
// ---------------------------------------------------------------------------

#[cfg(target_os = "macos")]
mod mach {
    extern "C" {
        pub fn mach_task_self() -> u32;
        pub fn mach_vm_region(
            target_task: u32,
            address: *mut u64,
            size: *mut u64,
            flavor: i32,
            info: *mut i32,
            info_cnt: *mut u32,
            object_name: *mut u32,
        ) -> i32;
        pub fn mach_vm_read_overwrite(
            target_task: u32,
            address: u64,
            size: u64,
            data: u64,
            out_size: *mut u64,
        ) -> i32;
    }
}

#[cfg(target_os = "macos")]
fn all_zero(buf: &[u8]) -> bool {
    // SAFETY: `align_to` only reinterprets the aligned middle part; every bit pattern is
    // a valid `u64`.
    let (pre, mid, post) = unsafe { buf.align_to::<u64>() };
    pre.iter().all(|&b| b == 0) && mid.iter().all(|&w| w == 0) && post.iter().all(|&b| b == 0)
}

/// Scan every readable+writable region of the current process for the signatures, at
/// every byte offset.
///
/// `sigs` is a flat buffer of `n * SIG_LEN` bytes. Hits that fall inside any of the
/// `exclude` address ranges (the caller's own signature storage) are ignored, as is the
/// scanner's internal copy buffer.
#[cfg(target_os = "macos")]
pub fn scan_self(sigs: &[u8], exclude: &[(usize, usize)]) -> Result<ScanReport, String> {
    if sigs.is_empty() || !sigs.len().is_multiple_of(SIG_LEN) {
        return Err("signature buffer must be a non-empty multiple of SIG_LEN".into());
    }
    let n = sigs.len() / SIG_LEN;
    let first_words: Vec<u32> = (0..n)
        .map(|i| u32::from_ne_bytes(sigs[i * SIG_LEN..i * SIG_LEN + 4].try_into().unwrap()))
        .collect();
    // Cheap pre-filter: only offsets whose first byte starts some signature are compared.
    let mut first_byte = [false; 256];
    for i in 0..n {
        first_byte[sigs[i * SIG_LEN] as usize] = true;
    }

    const CHUNK: usize = 1 << 20;
    // Consecutive chunks overlap by SIG_LEN bytes so a match straddling a boundary is
    // found exactly once.
    const ADVANCE: usize = CHUNK - SIG_LEN;

    let mut buf = vec![0u8; CHUNK];
    let buf_range = (buf.as_ptr() as usize, buf.as_ptr() as usize + CHUNK);
    let overlaps = |a: usize, b: usize, r: (usize, usize)| a < r.1 && r.0 < b;

    let mut report = ScanReport {
        hits: vec![0; n],
        ..Default::default()
    };

    // SAFETY: plain Mach calls on our own task with correctly sized out-parameters.
    let task = unsafe { mach::mach_task_self() };
    let mut addr: u64 = 0;
    loop {
        let mut size: u64 = 0;
        let mut info = [0i32; 9]; // vm_region_basic_info_64 is 9 ints
        let mut count: u32 = 9;
        let mut object_name: u32 = 0;
        let kr = unsafe {
            mach::mach_vm_region(
                task,
                &mut addr,
                &mut size,
                9, // VM_REGION_BASIC_INFO_64
                info.as_mut_ptr(),
                &mut count,
                &mut object_name,
            )
        };
        if kr != 0 || size == 0 {
            break; // KERN_INVALID_ADDRESS: past the last region
        }
        let region_end = addr + size;
        let protection = info[0];
        if protection & 3 == 3 {
            // readable and writable
            report.regions_scanned += 1;
            let mut p = addr;
            while p < region_end {
                let remaining = (region_end - p) as usize;
                let len = remaining.min(CHUNK);
                let last = len == remaining;
                let step = if last { len } else { ADVANCE };

                if !overlaps(p as usize, p as usize + len, buf_range) {
                    let mut out_size: u64 = 0;
                    let rc = unsafe {
                        mach::mach_vm_read_overwrite(
                            task,
                            p,
                            len as u64,
                            buf.as_mut_ptr() as u64,
                            &mut out_size,
                        )
                    };
                    if rc == 0 {
                        let got = out_size as usize;
                        report.bytes_scanned += got as u64;
                        let data = &buf[..got];
                        if !all_zero(data) {
                            // Start offsets this chunk is responsible for.
                            let limit = if last { got } else { ADVANCE }
                                .min(got.saturating_sub(SIG_LEN - 1));
                            for off in 0..limit {
                                if !first_byte[data[off] as usize] {
                                    continue;
                                }
                                let word = u32::from_ne_bytes(data[off..off + 4].try_into().unwrap());
                                for (i, &fw) in first_words.iter().enumerate() {
                                    if word == fw
                                        && data[off..off + SIG_LEN] == sigs[i * SIG_LEN..(i + 1) * SIG_LEN]
                                    {
                                        let hit = p as usize + off;
                                        let excluded = exclude
                                            .iter()
                                            .any(|&r| overlaps(hit, hit + SIG_LEN, r));
                                        if !excluded {
                                            report.hits[i] += 1;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                p += step as u64;
            }
        }
        addr = region_end;
    }
    Ok(report)
}

#[cfg(not(target_os = "macos"))]
pub fn scan_self(_sigs: &[u8], _exclude: &[(usize, usize)]) -> Result<ScanReport, String> {
    Err("process-memory scanning is only implemented for macOS".into())
}

pub fn is_supported() -> bool {
    cfg!(target_os = "macos")
}

// ---------------------------------------------------------------------------
// Scenarios
// ---------------------------------------------------------------------------

fn argmax(logits: &[f32]) -> u8 {
    let mut best = 0;
    for (i, &v) in logits.iter().enumerate() {
        if v > logits[best] {
            best = i;
        }
    }
    best as u8
}

fn generate_monolithic(model: &MonolithicSLM) {
    let mut ids = PROMPT.to_vec();
    for _ in 0..SCENARIO_TOKENS {
        let start = ids.len().saturating_sub(model.config.ctx_len);
        let logits = model.forward(&ids[start..]);
        ids.push(argmax(&logits));
    }
    std::hint::black_box(&ids);
}

fn generate_ephemeral(model: &mut EphemeralSLM) -> Result<(), String> {
    let telemetry = TelemetryLogger::new(false);
    let ctx_len = model.config().ctx_len;
    let mut ids = PROMPT.to_vec();
    for _ in 0..SCENARIO_TOKENS {
        let start = ids.len().saturating_sub(ctx_len);
        let logits = model
            .forward_ephemeral(&ids[start..], &telemetry)
            .map_err(|e| e.to_string())?;
        ids.push(argmax(&logits));
    }
    std::hint::black_box(&ids);
    Ok(())
}

fn err_string(e: SealError) -> String {
    e.to_string()
}

/// Decrypt every section into plain `Vec<u8>`s that nothing ever zeroes.
fn decrypt_into_plain_vecs(sealed: &SealedPayload, key: &SecretKey) -> Result<Vec<Vec<u8>>, String> {
    let mut sections = Vec::with_capacity(sealed.section_count());
    for i in 0..sealed.section_count() {
        let mut plain = vec![0u8; sealed.section_plain_len(i)];
        sealed.open_section(i, key, &mut plain).map_err(err_string)?;
        sections.push(plain);
    }
    Ok(sections)
}

/// Negative control: decrypt each section into a plain `Vec<u8>`, dequantize its tensors
/// into plain `Vec<f32>`s, and free everything without ever zeroing it. Shows what
/// "streaming without scrubbing" leaves behind, for both signature classes.
fn stream_without_scrub(key: &SecretKey) -> Result<(), String> {
    let sealed = SealedPayload::parse(EMBEDDED_SEALED_PAYLOAD).map_err(err_string)?;
    let last = sealed.section_count() - 1;
    for i in 0..sealed.section_count() {
        let mut plain = vec![0u8; sealed.section_plain_len(i)];
        sealed.open_section(i, key, &mut plain).map_err(err_string)?;
        {
            let tensors: Vec<QuantizedSlice> = if i == 0 {
                let (tok, pos) = parse_embedding_section(&plain);
                vec![tok, pos]
            } else if i == last {
                let (_, head) = parse_final_section(&plain);
                vec![head]
            } else {
                let s = parse_layer_section(&plain);
                vec![s.wq, s.wk, s.wv, s.wo, s.w1, s.w2]
            };
            for t in tensors {
                let mut v = vec![0.0_f32; t.num_elements()];
                t.dequantize_into(&mut v);
                std::hint::black_box(&v);
                drop(v);
            }
        }
        std::hint::black_box(&plain);
        drop(plain);
    }
    Ok(())
}

/// Run one scenario in this process and return the scan report. `sigs` must be the flat
/// signature buffer produced by [`from_hex`].
pub fn run_scenario(name: &str, sigs: &[u8], key: &SecretKey) -> Result<ScanReport, String> {
    let exclude = [(sigs.as_ptr() as usize, sigs.as_ptr() as usize + sigs.len())];
    match name {
        "monolithic-live" => {
            let model = MonolithicSLM::open(key).map_err(err_string)?;
            generate_monolithic(&model);
            let r = scan_self(sigs, &exclude);
            std::hint::black_box(&model);
            r
        }
        "plaintext-live" => {
            let sealed = SealedPayload::parse(EMBEDDED_SEALED_PAYLOAD).map_err(err_string)?;
            let held = decrypt_into_plain_vecs(&sealed, key)?;
            std::hint::black_box(&held);
            let r = scan_self(sigs, &exclude);
            std::hint::black_box(&held);
            r
        }
        "streaming-no-scrub" => {
            stream_without_scrub(key)?;
            scan_self(sigs, &exclude)
        }
        "monolithic-naive" => {
            let mut model = MonolithicSLM::open(key).map_err(err_string)?;
            generate_monolithic(&model);
            model.cleanup_naive();
            let r = scan_self(sigs, &exclude);
            std::hint::black_box(&model);
            r
        }
        "monolithic-naive-dropped" => {
            let mut model = MonolithicSLM::open(key).map_err(err_string)?;
            generate_monolithic(&model);
            model.cleanup_naive();
            drop(model);
            scan_self(sigs, &exclude)
        }
        "monolithic-secure" => {
            let mut model = MonolithicSLM::open(key).map_err(err_string)?;
            generate_monolithic(&model);
            model.cleanup();
            let r = scan_self(sigs, &exclude);
            std::hint::black_box(&model);
            r
        }
        "ephemeral" => {
            let mut model = EphemeralSLM::open(key).map_err(err_string)?;
            generate_ephemeral(&mut model)?;
            let r = scan_self(sigs, &exclude);
            std::hint::black_box(&model);
            r
        }
        other => Err(format!("unknown scenario {other:?}")),
    }
}

/// Entry point for a child process: load the key, decode signatures, run the scenario,
/// print one machine-readable `RESULT` line. `n_f32` says how many of the leading
/// signatures are f32 signatures (the rest are INT8).
pub fn run_child(scenario: &str, sig_hex: &str, n_f32: &str, key_file: Option<&str>) -> i32 {
    let key = match SecretKey::from_cli(key_file) {
        Ok(k) => k,
        Err(e) => {
            eprintln!("forensics child: {e}");
            return 2;
        }
    };
    let n_f32: usize = match n_f32.parse() {
        Ok(n) => n,
        Err(_) => {
            eprintln!("forensics child: bad signature split {n_f32:?}");
            return 2;
        }
    };
    let sigs = match from_hex(sig_hex) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("forensics child: bad signature hex: {e}");
            return 2;
        }
    };
    match run_scenario(scenario, &sigs, &key) {
        Ok(r) => {
            let split = n_f32.min(r.hits.len());
            let f32_hits: usize = r.hits[..split].iter().sum();
            let i8_hits: usize = r.hits[split..].iter().sum();
            let per_sig: Vec<String> = r.hits.iter().map(|h| h.to_string()).collect();
            println!(
                "RESULT scenario={} f32={} i8={} per_sig={} regions={} mib_scanned={:.1}",
                scenario,
                f32_hits,
                i8_hits,
                per_sig.join(","),
                r.regions_scanned,
                r.bytes_scanned as f64 / (1024.0 * 1024.0)
            );
            0
        }
        Err(e) => {
            eprintln!("forensics child: {e}");
            2
        }
    }
}

/// Parsed `RESULT` line from a child.
#[derive(Debug, Clone)]
pub struct ScenarioResult {
    pub scenario: String,
    pub f32_hits: usize,
    pub i8_hits: usize,
    pub per_sig: Vec<usize>,
    pub mib_scanned: f64,
}

fn parse_result(stdout: &str) -> Option<ScenarioResult> {
    let line = stdout.lines().find(|l| l.starts_with("RESULT "))?;
    let field = |key: &str| -> Option<&str> {
        line.split_whitespace()
            .find_map(|kv| kv.strip_prefix(key)?.strip_prefix('='))
    };
    Some(ScenarioResult {
        scenario: field("scenario")?.to_string(),
        f32_hits: field("f32")?.parse().ok()?,
        i8_hits: field("i8")?.parse().ok()?,
        per_sig: field("per_sig")?
            .split(',')
            .filter_map(|s| s.parse().ok())
            .collect(),
        mib_scanned: field("mib_scanned")?.parse().ok()?,
    })
}

/// Parent entry point: decrypt once to compute signatures, run every scenario in a fresh
/// child process, print a table, and return a process exit code (0 = all assertions held).
/// `key_file` is passed on to the children (they otherwise inherit `ESLM_KEY`).
pub fn run_parent(key_file: Option<&str>) -> i32 {
    if !is_supported() {
        println!("[forensics] process-memory scanning is only implemented for macOS; skipping.");
        return 0;
    }
    let key = match SecretKey::from_cli(key_file) {
        Ok(k) => k,
        Err(e) => {
            eprintln!("[forensics] {e}");
            return 2;
        }
    };
    let sigs = match compute_signatures(&key) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("[forensics] cannot build signatures: {e}");
            return 2;
        }
    };
    drop(key);
    let n_f32 = sigs.iter().filter(|s| s.kind == SigKind::F32).count();
    let flat: Vec<u8> = sigs.iter().flat_map(|s| s.bytes).collect();
    let hex = to_hex(&flat);
    let exe = match std::env::current_exe() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("[forensics] cannot locate own executable: {e}");
            return 2;
        }
    };

    println!(
        "[forensics] {} signatures of {} bytes: {} decompressed f32 + {} decrypted INT8",
        sigs.len(),
        SIG_LEN,
        n_f32,
        sigs.len() - n_f32
    );
    println!(
        "            tensors: {}",
        sigs.iter()
            .filter(|s| s.kind == SigKind::F32)
            .map(|s| s.label.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!("[forensics] one child process per scenario; scanning all rw- regions at every byte offset\n");
    println!("{:<26} {:<12} {:<12} verdict", "scenario", "f32 weights", "int8 plain");
    println!("{}", "-".repeat(96));

    let mut failures = 0;
    for sc in SCENARIOS {
        let mut cmd = std::process::Command::new(&exe);
        cmd.args(["--forensics-child", sc.name, &hex, &n_f32.to_string()]);
        if let Some(path) = key_file {
            cmd.args(["--key-file", path]);
        }
        let result = match cmd.output() {
            Ok(o) if o.status.success() => parse_result(&String::from_utf8_lossy(&o.stdout)),
            Ok(o) => {
                eprintln!("child {} failed: {}", sc.name, String::from_utf8_lossy(&o.stderr));
                None
            }
            Err(e) => {
                eprintln!("could not spawn child {}: {e}", sc.name);
                None
            }
        };
        let Some(r) = result else {
            println!("{:<26} {:<12} {:<12} ERROR", sc.name, "-", "-");
            failures += 1;
            continue;
        };
        let ok = sc.f32_expect.holds(r.f32_hits) && sc.i8_expect.holds(r.i8_hits);
        if !ok {
            failures += 1;
        }
        let both_info = sc.f32_expect == Expect::Informational && sc.i8_expect == Expect::Informational;
        let verdict = match (ok, both_info) {
            (false, _) => "FAIL".to_string(),
            (true, true) => format!("(reported) {}", sc.desc),
            (true, false) => format!("ok   {}", sc.desc),
        };
        println!(
            "{:<26} {:<12} {:<12} {}",
            r.scenario,
            format!("{:>3} {}", r.f32_hits, sc.f32_expect.tag()),
            format!("{:>3} {}", r.i8_hits, sc.i8_expect.tag()),
            verdict
        );
    }
    println!("\n(>0 = positive control, must find; =0 = must find nothing; info = reported only)");
    if failures == 0 {
        println!("[forensics] all assertions held");
        0
    } else {
        println!("[forensics] {failures} assertion(s) FAILED");
        1
    }
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;

    fn demo_key() -> SecretKey {
        SecretKey::from_file(concat!(env!("CARGO_MANIFEST_DIR"), "/assets/demo.key"))
            .expect("assets/demo.key must exist for the tests")
    }

    #[test]
    fn hex_roundtrip() {
        let bytes: Vec<u8> = (0..=255).collect();
        assert_eq!(from_hex(&to_hex(&bytes)).unwrap(), bytes);
        assert!(from_hex("abc").is_err());
        assert!(from_hex("zz").is_err());
    }

    #[test]
    fn signatures_are_distinct_and_nonzero() {
        let sigs = compute_signatures(&demo_key()).unwrap();
        assert_eq!(sigs.iter().filter(|s| s.kind == SigKind::F32).count(), 7);
        assert_eq!(sigs.iter().filter(|s| s.kind == SigKind::Int8).count(), 7);
        // f32 signatures must come first: the child splits the hit list by position.
        let first_i8 = sigs.iter().position(|s| s.kind == SigKind::Int8).unwrap();
        assert!(sigs[first_i8..].iter().all(|s| s.kind == SigKind::Int8));
        for s in &sigs {
            assert!(s.bytes.iter().any(|&b| b != 0), "{} is all zero", s.label);
        }
        let mut uniq: Vec<[u8; SIG_LEN]> = sigs.iter().map(|s| s.bytes).collect();
        uniq.sort();
        uniq.dedup();
        assert_eq!(uniq.len(), sigs.len());
    }

    /// The scanner must see known patterns in the heap at *any* byte offset, and stop seeing
    /// them after they are zeroed. Uses a delta so unrelated stack copies of the pattern
    /// cannot interfere.
    #[test]
    fn scanner_finds_and_loses_planted_patterns_at_any_alignment() {
        // A pattern that does not occur anywhere else.
        let mut sig = Vec::with_capacity(SIG_LEN);
        for i in 0..SIG_LEN as u8 {
            sig.push(i.wrapping_mul(37).wrapping_add(0xA5) ^ 0x5C);
        }
        let exclude = [(sig.as_ptr() as usize, sig.as_ptr() as usize + SIG_LEN)];

        // One aligned copy and two deliberately misaligned ones.
        let mut planted = vec![0u8; 4096];
        for offset in [100, 201, 302] {
            planted[offset..offset + SIG_LEN].copy_from_slice(&sig);
        }
        std::hint::black_box(&planted);

        let with = scan_self(&sig, &exclude).unwrap();
        assert!(with.bytes_scanned > 0 && with.regions_scanned > 0);

        crate::ephemeral::volatile_zero(&mut planted);
        std::hint::black_box(&planted);
        let without = scan_self(&sig, &exclude).unwrap();

        assert_eq!(
            with.total_hits(),
            without.total_hits() + 3,
            "scanner should lose exactly the three planted copies"
        );
    }
}
