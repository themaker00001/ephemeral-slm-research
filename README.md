# Ephemeral SLM Research - Concrete Implementation

An empirical Rust implementation and security research testbed demonstrating the mechanics of **Ephemeral Small Language Models (SLMs)**:

1. **Source-Embedded Compressed Parameters**: 50,928 model weights stored as symmetric INT8 quantized arrays embedded directly into the binary's `.rodata` section via `include_bytes!`. No external weight files (`.bin`, `.gguf`) are required.
2. **Transient Layer Streaming ("Ephemeral Execution")**: Rather than maintaining the entire model in RAM, weights are reconstituted on-the-fly layer-by-layer during inference and immediately eradicated.
3. **Guaranteed Volatile Zeroization**: Uses `core::ptr::write_volatile` paired with `std::sync::atomic::compiler_fence(Ordering::SeqCst)` to strictly prohibit LLVM dead-code elimination (DCE) from optimizing away buffer cleanup.
4. **Live OS Kernel Telemetry**: Real-time Resident Set Size (RSS) monitoring directly via macOS Mach kernel APIs (`mach_task_basic_info`) and Linux `/proc/self/statm`.
5. **Memory Forensic Verification**: Performs byte-by-byte heap audits confirming that 100% of memory buffers are zeroed (`0x00`) with zero parameter residue after execution.

---

## Architecture

The model is an autoregressive character/byte-level micro-transformer (LLaMA-style):
- **Vocabulary**: 128 ASCII tokens (character/byte-level generation)
- **Context Length**: 32 tokens
- **Hidden Dimension ($d_{\text{model}}$)**: 48
- **Layers**: 2 Transformer Blocks
- **Attention Heads**: 3 ($d_{\text{head}} = 16$, scaled dot-product causal attention)
- **FeedForward**: $d_{\text{ff}} = 96$ with SiLU activation
- **Normalization**: Root Mean Square Normalization (RMSNorm) with learnable scale $\gamma$
- **Total Weights**: 50,928 parameters (~51 KB INT8 binary payload)

---

## Monolithic vs. Ephemeral Execution

```
                     MONOLITHIC INFERENCE (Baseline)
               +---------------------------------------+
   Process RAM | [Embeddings] [Layer 0] [Layer 1] [Head]|  <- 198 KB resident
               +---------------------------------------+
                     Weights stay resident for process lifetime.

                     EPHEMERAL STREAMING INFERENCE
   Step 1: Reconstruct Embeddings -> Lookup -> [SECURE SCRUB] (0 KB left)
   Step 2: Reconstruct Layer 0   -> Forward -> [SECURE SCRUB] (0 KB left)
   Step 3: Reconstruct Layer 1   -> Forward -> [SECURE SCRUB] (0 KB left)
   Step 4: Reconstruct Head      -> Logits  -> [SECURE SCRUB] (0 KB left)
               +---------------------------------------+
   Process RAM |       [Active Layer Only: ~72 KB]     |  <- Only 1 layer in RAM!
               +---------------------------------------+
                     Between steps & post-run: 0 KB model weights in RAM.
```

---

## Quick Start

### 1. Run Ephemeral Generation (Default)
```bash
cargo run -- --prompt "The ephemeral " --tokens 30
```

### 2. Run Verbose Lifecycle Telemetry
See exact microsecond reconstruction, volatile scrub timing, and OS-level RSS changes:
```bash
cargo run -- --prompt "Safe " --tokens 5 -v
```

### 3. Compare Monolithic vs Ephemeral in Benchmark
```bash
cargo run --release -- --benchmark
```

Sample Benchmark Output:
```text
========================== RESULTS ==============================
Metric                     | Monolithic        | Ephemeral
---------------------------+-------------------+------------------
Active Param Residency     |           198 KB |            72 KB (-63%)
Post-Run Memory Footprint  |           198 KB |             0 KB (100% wiped)
Total Bytes Wiped          |        203,712 B  |     7,129,920 B 
Volatile Memory Audits     |           N/A     |            140 OK
Init / Load Latency        |         212 µs    |            2.3 µs
Generation Latency         |        26.8 ms    |           20.1 ms
Per-Token Latency          |      767 µs /t    |         573 µs /t
Output Text Parity         | 100% IDENTICAL (EXACT MATCH)
==================================================================
```

### 4. Run Memory Zeroization Forensics Audit
```bash
cargo run -- --verify
```

---

## Security Mechanics: Why `Vec::fill(0.0)` is Insufficient

In standard Rust, code like:
```rust
buffer.fill(0.0);
drop(buffer);
```
can be completely eliminated by the LLVM compiler during optimization passes because the buffer is dropped immediately afterward (Dead Code Elimination). 

The `SecureBuffer<T>` implementation solves this with volatile pointer writes and sequential consistency barriers:

```rust
// 1. Acquire barrier
compiler_fence(Ordering::SeqCst);

// 2. Volatile write guarantees CPU store instructions are physically emitted
unsafe {
    for i in 0..total_bytes {
        std::ptr::write_volatile(ptr.add(i), 0u8);
    }
}

// 3. Release barrier prevents compiler from hoisting or reordering
compiler_fence(Ordering::SeqCst);
```

---

## Project Structure

- [`src/weights.rs`](src/weights.rs): INT8 quantized payload parser and dequantization routines.
- [`src/ephemeral.rs`](src/ephemeral.rs): `SecureBuffer<T>` implementation, volatile scrubbers, and forensic audit checks.
- [`src/telemetry.rs`](src/telemetry.rs): macOS Mach kernel task info queries and structured event emission.
- [`src/model.rs`](src/model.rs): Autoregressive micro-transformer forward pass (RMSNorm, MHA, SiLU FFN).
- [`src/main.rs`](src/main.rs): CLI flags, benchmark harness, and interactive generation.
- [`assets/weights.bin`](assets/weights.bin): Packed 51,820 byte INT8 model parameters.
- [`scripts/train_export.py`](scripts/train_export.py): PyTorch training script to retrain or export custom weights.
- [`tests/integration_tests.rs`](tests/integration_tests.rs): Mathematical equivalence and memory scrubbing tests.

---

## License & Defensive Research Scope

This testbed is designed for defensive AI security research, process memory profiling, and edge/enclave memory reduction. It contains no exploit payloads, network functionality, or persistence mechanisms.
