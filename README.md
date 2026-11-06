# Ephemeral SLM Research

A small Rust testbed (no dependencies) for one narrow question:

> If a tiny language model is stored compressed inside the binary, and its decompressed
> weights are rebuilt one layer at a time and wiped right after use, how much decompressed
> weight data is ever sitting in writable memory, and can we *measure* that nothing is left over?

It is a research toy: a 50,928-parameter byte-level transformer trained for a few hundred
steps on a ~700-character corpus. Its output is gibberish; the point is the memory mechanics.

## What this does and does not show

**It shows**

- The decompressed (f32) weights can be kept to one layer at a time (peak 72 KB instead of
  199 KB), at a measured cost of roughly 1.5x generation time.
- A volatile-write scrub removes the decompressed weights from process memory: a scan of the
  whole writable address space finds **no** copy afterwards, while a positive control proves
  the scanner can find them when they are there (see [Forensics](#forensics-did-anything-survive)).

**It does not show**

- **Model confidentiality.** The INT8 payload (`assets/weights.bin`, 50.6 KB) is embedded in
  the binary and stays resident for the whole run in *both* engines. Anyone with the
  executable has the entire model. "Ephemeral" here only describes the *decompressed working
  copies*. The technique would only protect a model if the compressed form were itself
  protected (for example encrypted and decrypted per layer). That is not implemented.
- Anything about registers, swap, hibernation images, other processes, or hardware.
- A speed or efficiency win. At this size it is a pure cost (see the benchmark).

## Architecture

Autoregressive byte-level micro-transformer (LLaMA-style):

- Vocabulary 128 (ASCII), context 32, `d_model` 48, 2 layers, 3 heads (`d_head` 16)
- Feed-forward `d_ff` 96 with SiLU, RMSNorm with learnable scale
- 50,928 parameters; weights are symmetric INT8 with one f32 scale per tensor (norm vectors
  stay f32). Packed payload: 51,820 bytes including headers.

Two engines run the same math:

| | Monolithic (baseline) | Ephemeral |
|---|---|---|
| Decompressed weights | all dequantized once, resident for the whole run | one layer at a time, rebuilt per token |
| After use | one volatile wipe at the end | volatile wipe + zero-audit after every layer |
| Activations / scratch (`x`, Q/K/V, attention scores, FFN temporaries) | Q/K/V, scores and FFN scratch are zeroed on drop; `x` is a plain `Vec` | all of them zeroed on drop |

Not scrubbed in either engine: the returned logits and the generated tokens (they are the
output), and stack temporaries and registers.

## Quick start

```bash
cargo run --release -- --prompt "The ephemeral " --tokens 30   # generate
cargo run --release -- --prompt "Safe " --tokens 5 -v          # per-layer lifecycle telemetry
cargo run --release -- --benchmark                             # monolithic vs ephemeral
cargo run --release -- --forensics                             # scan process memory (macOS)
cargo run --release -- --verify                                # check SecureBuffer zeroing only
cargo test                                                     # 10 tests, includes the forensics run
```

## Benchmark

Measured on an Apple M4 (macOS 26, arm64), release build, 35 tokens, median of 21 runs after
3 warm-ups, with the run order alternating each iteration. Both engines use the *same*
volatile wipe, so only residency differs.

```text
Metric                              | Monolithic         | Ephemeral
------------------------------------+--------------------+-----------------
Decompressed weights, peak resident |           198.9 KB |         72.4 KB
Decompressed weights after run      |  198.9 KB (zeroed) |          0.0 KB
INT8 payload in .rodata (always)    |            50.6 KB |         50.6 KB
Bytes volatile-wiped per run        |             203712 |         7129920
Zero-audits passed / failed         |                n/a |         140 / 0
Upfront cost (median)               |             5.12µs |        167.00ns
Wipe cost, explicit (median)        |            49.71µs |   in generation
Generation (median)                 |             7.60ms |         11.25ms
Generation (min .. max)             |       7.2ms..7.8ms |  10.8ms..11.4ms
Per token (median)                  |           217.08µs |        321.38µs
Init + generation + wipe (median)   |             7.65ms |         11.25ms
Output parity (all runs)            | IDENTICAL
```

Reading it honestly:

- Peak decompressed-weight residency drops by about 64%.
- Ephemeral is about **1.5x slower** (the min..max ranges do not overlap). The likely cause is
  the wipe: the explicit-wipe row shows about 50 µs per 200 KB, and the ephemeral engine wipes
  roughly that much on every token and then reads it back for the audit, which is about the
  size of the ~104 µs/token gap. I did not profile this separately.
- Output is bit-for-bit identical between engines on every run.
- An earlier version of this README showed ephemeral as *faster*. That was an artifact of
  running the baseline first and cold, with no warm-up, and of comparing different things in
  the "init" row. The numbers above replace it. Your numbers will differ; use `--reps`.

## Forensics: did anything survive?

`SecureBuffer::audit_is_zero` only re-reads the buffer that was just overwritten, so it can
only confirm that the zeroing loop ran. It cannot see copies elsewhere. `--forensics` answers
that question by scanning memory instead:

1. The parent computes seven 32-byte signatures of **decompressed** f32 weights (8 consecutive
   values from the middle of seven tensors).
2. For each scenario it starts a **fresh child process**, passes the signatures as ASCII hex
   (so the child holds the raw bytes in exactly one buffer, which is excluded from the scan),
   runs the scenario, and then walks every readable+writable region of its own address space
   (`mach_vm_region` + `mach_vm_read_overwrite`), counting signature hits.

Result on the same machine (identical across 5 consecutive release runs and in a debug build):

| Scenario | Hits | Meaning |
|---|---:|---|
| `monolithic-live` | 7 | **Positive control.** Weights are resident, the scanner finds all 7 signatures. |
| `streaming-no-scrub` | 3 | **Negative control.** Streaming tensors through plain `Vec`s and freeing them leaves copies in freed memory. Streaming alone does not help; the scrub does. |
| `monolithic-naive` | 0 | Plain `fill(0.0)` cleanup, model still allocated. |
| `monolithic-naive-dropped` | 0 | Plain `fill(0.0)` cleanup, then the model is dropped. |
| `monolithic-secure` | 0 | Volatile cleanup. |
| `ephemeral` | 0 | Layer streaming plus volatile scrub, after 20 tokens. |

Only the control, secure and ephemeral rows are asserted (by `cargo test`); the others are
reported because they depend on the compiler and allocator.

Sanity check that the assertions can fail: with `volatile_zero` stubbed out to do nothing,
`monolithic-secure` finds 7 hits, `ephemeral` finds 4, and the run exits non-zero.

Limits of this experiment:

- macOS only. Other platforms skip it and report that it is unsupported.
- Seven signatures of 32 bytes at 4-byte alignment: a partial or shifted copy would be missed.
- It does not look at registers, swap, core dumps or hibernation files, or at other processes.
- It scans a 20-token run in a small process; a long-running server would behave differently.

## Why a volatile scrub, if plain `fill(0.0)` also worked here?

The Rust and LLVM memory model allows a compiler to delete a store to memory that is about to
be freed, so `buffer.fill(0.0); drop(buffer);` carries no guarantee. **In this build it was not
deleted**: `monolithic-naive` and `monolithic-naive-dropped` both come back clean on rustc 1.95
in release and debug. So the volatile primitive here is defence against what a compiler is
*permitted* to do, not a fix for something observed. `volatile_zero` does the wipe with
`write_volatile` plus `compiler_fence(SeqCst)`; it covers the bytes of the slice it is given
and nothing else.

## Project structure

- [`src/weights.rs`](src/weights.rs): INT8 payload parser and dequantisation. The payload comes
  from `include_bytes!`, which is only 1-byte aligned, so f32 data is decoded with
  `f32::from_le_bytes` rather than reinterpreted in place (an earlier version did the latter,
  which is undefined behaviour and aborted in debug builds on a misaligned payload).
- [`src/ephemeral.rs`](src/ephemeral.rs): `volatile_zero`, `SecureBuffer<T>`, scrub metrics.
- [`src/model.rs`](src/model.rs): the two engines (RMSNorm, multi-head attention, SiLU FFN).
- [`src/telemetry.rs`](src/telemetry.rs): RSS via Mach `task_info` (macOS) or `/proc` (Linux,
  not run by the author), plus lifecycle events.
- [`src/forensics.rs`](src/forensics.rs): signature building, the memory scanner, the scenarios.
- [`src/main.rs`](src/main.rs): CLI, benchmark harness.
- [`assets/weights.bin`](assets/weights.bin): the packed INT8 model.
- [`scripts/train_export.py`](scripts/train_export.py): PyTorch script that trains and exports weights.
- [`tests/integration_tests.rs`](tests/integration_tests.rs): engine parity, scrubbing, alignment
  regression, and the end-to-end forensics run.

## Scope

Defensive research on memory residency and observability. It contains no exploit payloads,
network code, persistence, or self-deletion. Only run it on machines you own.
