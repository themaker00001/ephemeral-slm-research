# Ephemeral SLM Research

A small Rust testbed (no dependencies) for one narrow question:

> If a tiny language model is stored **encrypted** inside the binary, and each layer is
> decrypted and decompressed only when it is needed and wiped right after use, how much of
> the model is ever sitting in memory in the clear, and can we *measure* that nothing is
> left over?

It is a research toy: a 50,928-parameter byte-level transformer trained for a few hundred
steps on a ~700-character corpus. Its output is gibberish; the point is the memory mechanics.

## What this does and does not show

**It shows**

- The executable carries the model only as ciphertext (ChaCha20-Poly1305, one authenticated
  section per layer). Counting 32-byte pieces of the plaintext weights inside the built
  executable: the previous version contained **1,618 of 1,618**; this one contains **0**.
- The decompressed (f32) weights can be kept to one layer at a time (peak 72 KB instead of
  199 KB), and each layer's decrypted INT8 plaintext is wiped as soon as it has been
  dequantized.
- A volatile-write scrub removes both kinds of plaintext from process memory: a byte-by-byte
  scan of the whole writable address space finds **no** copy afterwards, while positive
  controls prove the scanner can find them when they are there (see
  [Forensics](#forensics-did-anything-survive)).

**It does not show**

- **Protection against anyone who can run the program or read its memory.** The key has to
  be in the process while the model runs, and each layer is in the clear for a moment. See
  [Threat model](#threat-model).
- Anything about registers, swap, hibernation images, other processes, or hardware.
- A speed or efficiency win. At this size it is a pure cost (see the benchmark).
- Production-grade cryptography. The cipher is written from scratch to keep the crate
  dependency-free; see [Threat model](#threat-model).

## Architecture

Autoregressive byte-level micro-transformer (LLaMA-style):

- Vocabulary 128 (ASCII), context 32, `d_model` 48, 2 layers, 3 heads (`d_head` 16)
- Feed-forward `d_ff` 96 with SiLU, RMSNorm with learnable scale
- 50,928 parameters; weights are symmetric INT8 with one f32 scale per tensor (norm vectors
  stay f32)

The model is embedded as a **sealed** file (`assets/weights.enc`, 51,916 bytes): a small
header plus four sections (embeddings, layer 0, layer 1, final norm + head), each encrypted
and authenticated separately. Section `i` uses nonce `prefix || i` and authenticates the
whole header plus `i`, so editing the header, reordering sections or flipping any bit is
detected. The key is not in the executable; it comes from `--key-file` or `ESLM_KEY`.

Two engines run the same math:

| | Monolithic (baseline) | Ephemeral |
|---|---|---|
| Decrypted INT8 | one section at a time, wiped after dequantizing, once at load | one section at a time, wiped after dequantizing, on every token |
| Decompressed f32 weights | all dequantized once, resident for the whole run | one layer at a time, rebuilt per token |
| After use | one volatile wipe at the end | volatile wipe + zero-audit after every layer |
| Integrity check | each section's tag, when it is opened | every tag when the model is opened, and again per section |
| Activations / scratch (`x`, Q/K/V, attention scores, FFN temporaries) | Q/K/V, scores and FFN scratch are zeroed on drop; `x` is a plain `Vec` | all of them zeroed on drop |

Not scrubbed in either engine: the returned logits and the generated tokens (they are the
output), the key (it lives for the run), and stack temporaries and registers.

## Threat model

| Who | Can they get the weights? |
|---|---|
| Someone with a copy of the executable only | **No** (without the key). This is what sealing adds. Before it, the weights were plain bytes in the file. |
| Someone who modifies the embedded model | Detected: the tags fail, and the engines refuse to run. |
| Someone with the executable **and** the key | Yes, trivially. |
| Someone who can read the running process (debugger, memory dump) | Yes. The key stays in memory for the whole run (both engines), and each layer is in the clear while it is being used. The per-layer scheme shortens that window; it does not close it. |
| Someone who can read the key file or the environment | Yes. Prefer a key file with mode 0600; an environment variable is readable by other processes of the same user. |

Other things to know:

- **The demo key is public.** `assets/demo.key` is committed so the quick start and the tests
  run, and the plaintext `assets/weights.bin` (the training output, used as a test
  reference) is committed too. The demo model has no secret. The mechanism is the point.
- **The cipher is hand-written.** ChaCha20-Poly1305 (RFC 8439) in `src/seal.rs`. It passes
  the RFC test vectors, matches Python's `cryptography` package on 13 message lengths, and
  the committed `weights.enc` was produced by that package, so the Rust decryptor is checked
  against a second implementation. It has not been audited and has no side-channel
  hardening beyond avoiding secret-dependent branches and indices. For anything real, use a
  vetted crate.
- One key is used for every section (the nonces differ); there is no key derivation from a
  passphrase. The nonce prefix is 64 random bits per seal, so do not seal two different
  payloads under the same key and the same prefix.
- The architecture (layer count, widths) is stored in the clear in the header. It is
  authenticated, not secret.
- The scan below does not look for the key itself; that it stays in memory is by design.

## Quick start

```bash
KEY=assets/demo.key
cargo run --release -- --key-file $KEY --prompt "The ephemeral " --tokens 30   # generate
cargo run --release -- --key-file $KEY --prompt "Safe " --tokens 5 -v          # per-layer telemetry
cargo run --release -- --key-file $KEY --benchmark                             # monolithic vs ephemeral
cargo run --release -- --key-file $KEY --forensics                             # scan process memory (macOS)
cargo run --release -- --verify                                                # SecureBuffer check only, no key
cargo test                                                                     # 30 tests, includes the forensics run
```

With no key, or the wrong one, the program exits with status 2 before doing any work.

### Sealing your own model

```bash
python3 scripts/train_export.py                                   # trains, writes assets/weights.bin (plaintext)
cargo run --release -- --gen-key my.key                           # new random key, mode 0600, never overwrites
cargo run --release -- --seal assets/weights.bin assets/weights.enc --key-file my.key --force
cargo build --release                                             # embeds the new weights.enc
```

`--seal` reads the result back and checks it opens to the input before writing. The tests
expect `assets/weights.enc` to open with `assets/demo.key`, so they will fail after you
reseal under a different key. `scripts/seal_model.py` is an independent sealer and opener
(`pip install cryptography`) for the same format; the committed asset was made with it.

## Benchmark

Measured on an Apple M4 (macOS 26, arm64), release build, 35 tokens, median of 21 runs after
3 warm-ups, with the run order alternating each iteration. Both engines decrypt from the
same sealed payload and use the *same* volatile wipe, so only residency and how often they
decrypt differ.

```text
Metric                              | Monolithic         | Ephemeral
------------------------------------+--------------------+-----------------
Decompressed weights, peak resident |           198.9 KB |         72.4 KB
Decompressed weights after run      |  198.9 KB (zeroed) |          0.0 KB
Sealed payload in .rodata (always)  |            50.7 KB |         50.7 KB
Decrypted INT8, most held at once   |            18.4 KB |         18.4 KB
Decrypted INT8 wiped per run        |              51788 |         1812580
Decompressed bytes wiped per run    |             203712 |         7129920
Zero-audits passed / failed         |                n/a |         280 / 0
Upfront cost incl. decrypt (median) |           128.79µs |         26.17µs
Wipe cost, explicit (median)        |            50.83µs |   in generation
Generation (median)                 |             7.80ms |         16.47ms
Generation (min .. max)             |       7.3ms..7.9ms |  16.0ms..16.7ms
Per token (median)                  |           222.73µs |        470.60µs
Init + generation + wipe (median)   |             7.98ms |         16.50ms
Output parity (all runs)            | IDENTICAL
```

Reading it honestly:

- Peak decompressed-weight residency drops by about 64%.
- Ephemeral is about **2.1x slower** than monolithic (the min..max ranges do not overlap).
  Before encryption was added the same comparison was about 1.5x (7.6 ms vs 11.3 ms), so
  sealing cost the ephemeral engine roughly 5 ms over 35 tokens. Decrypting and
  authenticating every section once takes about 115 µs, and the ephemeral engine does that
  on every token, which accounts for about 4 ms. The rest is probably the extra wipe and
  audit of 1.8 MB of decrypted INT8; I did not profile it separately.
- The monolithic engine pays for decryption once, in its upfront cost.
- Output is bit-for-bit identical between engines on every run, and identical to the
  unencrypted version.
- Your numbers will differ; use `--reps`.

## Forensics: did anything survive?

`SecureBuffer::audit_is_zero` only re-reads the buffer that was just overwritten, so it can
only confirm that the zeroing loop ran. It cannot see copies elsewhere. `--forensics`
answers that question by scanning memory instead:

1. The parent decrypts the model once and computes fourteen 32-byte signatures: seven of
   **decompressed f32** weights (8 consecutive values from the middle of seven tensors) and
   seven of **decrypted INT8** weights (32 consecutive values from the same tensors).
2. For each scenario it starts a **fresh child process**, passes the signatures as ASCII hex
   (so the child holds the raw bytes in exactly one buffer, which is excluded from the scan),
   runs the scenario, and then walks every readable+writable region of its own address space
   (`mach_vm_region` + `mach_vm_read_overwrite`), counting signature hits at **every byte
   offset**.

Result on the same machine (identical across 5 consecutive release runs and in a debug build):

| Scenario | f32 hits | INT8 hits | Meaning |
|---|---:|---:|---|
| `monolithic-live` | 7 | 0 | **Positive control (f32).** All weights resident, the scanner finds all 7. The decrypted INT8 was already wiped at load. |
| `plaintext-live` | 0 | 7 | **Positive control (INT8).** Every section decrypted into plain `Vec`s and kept: the scanner finds all 7. |
| `streaming-no-scrub` | 3 | 2 | **Negative control.** Decrypting and dequantizing through plain `Vec`s and freeing them leaves copies of both kinds in freed memory. Streaming alone does not help; the scrub does. |
| `monolithic-naive` | 0 | 0 | Plain `fill(0.0)` cleanup, model still allocated. |
| `monolithic-naive-dropped` | 0 | 0 | Plain `fill(0.0)` cleanup, then the model is dropped. |
| `monolithic-secure` | 0 | 0 | Volatile cleanup. |
| `ephemeral` | 0 | 0 | Layer decrypt plus volatile scrub, after 20 tokens. |

The two controls and the clean scenarios are asserted (by `cargo test`); `streaming-no-scrub`
and the f32 column of the naive rows are reported because they depend on the compiler and
allocator.

Sanity check that the assertions can fail: with `volatile_zero` stubbed out to do nothing,
`monolithic-secure` finds 7 f32 and 2 INT8 copies, `ephemeral` finds 4 f32 and 5 INT8, and
the run exits non-zero (5 scenarios fail).

Limits of this experiment:

- macOS only. Other platforms skip it and report that it is unsupported.
- Seven signatures of each kind, 32 bytes each: a partial copy shorter than 32 bytes would be
  missed.
- It does not look at registers, swap, core dumps or hibernation files, or at other
  processes, and it does not scan for the key.
- It scans a 20-token run in a small process; a long-running server would behave differently.

## Why a volatile scrub, if plain `fill(0.0)` also worked here?

The Rust and LLVM memory model allows a compiler to delete a store to memory that is about to
be freed, so `buffer.fill(0.0); drop(buffer);` carries no guarantee. **In this build it was not
deleted**: `monolithic-naive` and `monolithic-naive-dropped` both come back clean on rustc 1.95
in release and debug. So the volatile primitive here is defence against what a compiler is
*permitted* to do, not a fix for something observed. `volatile_zero` does the wipe with
`write_volatile` plus `compiler_fence(SeqCst)`; it covers the bytes of the slice it is given
and nothing else. (The decrypted-INT8 buffers, the key, and the cipher's working state use
it too.)

## Project structure

- [`src/seal.rs`](src/seal.rs): ChaCha20-Poly1305 (RFC 8439), the sealed container, key
  loading, `seal_payload`. Unit tests include the RFC vectors and tamper cases.
- [`src/weights.rs`](src/weights.rs): the embedded sealed payload, INT8 dequantisation, and the
  parsers for a decrypted section. Fields are decoded with `f32::from_le_bytes` rather than
  reinterpreted in place (an earlier version did the latter, which is undefined behaviour on a
  misaligned payload).
- [`src/ephemeral.rs`](src/ephemeral.rs): `volatile_zero`, `SecureBuffer<T>`, scrub metrics.
- [`src/model.rs`](src/model.rs): the two engines (RMSNorm, multi-head attention, SiLU FFN).
- [`src/telemetry.rs`](src/telemetry.rs): RSS via Mach `task_info` (macOS) or `/proc` (Linux,
  not run by the author), plus lifecycle events.
- [`src/forensics.rs`](src/forensics.rs): signature building, the memory scanner, the scenarios.
- [`src/main.rs`](src/main.rs): CLI (`--gen-key`, `--seal`, benchmark harness).
- [`assets/weights.enc`](assets/weights.enc): the sealed model that is embedded.
- [`assets/demo.key`](assets/demo.key): the public demo key.
- [`assets/weights.bin`](assets/weights.bin): the plaintext training output, kept only as a
  test reference; it is not embedded in the executable.
- [`scripts/train_export.py`](scripts/train_export.py): PyTorch script that trains and exports weights.
- [`scripts/seal_model.py`](scripts/seal_model.py): independent sealer/opener built on `cryptography`.
- [`tests/integration_tests.rs`](tests/integration_tests.rs): engine parity, sealing and tamper
  rejection, alignment regression, "no plaintext in the executable", the CLI, and the
  end-to-end forensics run.

## Scope

Defensive research on memory residency and observability. It contains no exploit payloads,
network code, persistence, or self-deletion. Only run it on machines you own.
