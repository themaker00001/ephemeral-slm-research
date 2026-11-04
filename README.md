# Ephemeral SLM Research Demo

A small, safe Rust experiment demonstrating the *mechanics* of:

1. Keeping toy model parameters as constants in source code.
2. Reconstructing floating-point tensors at runtime.
3. Running a benign classification inference.
4. Emitting observable lifecycle events.
5. Overwriting the model's owned buffers before normal process exit.

This is deliberately **not** an attack agent, malware loader, stealth mechanism,
or self-deleting program. It does not hide execution, evade monitoring, persist,
access a network, or remove its own files.

## Run

Install Rust, then:

```bash
cargo run
```

## What to observe

The program prints lifecycle events to stderr:

- `model_reconstructed`
- `cleanup_requested`

The model is a tiny linear classifier with two input features and three output
classes. Its six weights and three biases are synthetic values, quantized as
small integers and dequantized at runtime.

## Important security and memory limitations

- Source-embedded constants are obfuscation at most, not encryption. Anyone with
  the executable or source can recover them.
- A real multi-billion-parameter model is far too large for this toy
  representation to be practical.
- `Vec::fill(0.0)` is only a best-effort overwrite. It does not guarantee that
  every copy has been erased from registers, compiler temporaries, allocator
  pages, swap, hibernation images, or crash dumps.
- The process exits normally and leaves its source and executable files intact.

## Suggested defensive experiments

- Record process start/exit times and memory usage with OS-level monitoring.
- Compare source-embedded constants with a conventional model file.
- Add a detector that flags unusually large numeric constant arrays.
- Measure peak resident memory during reconstruction and cleanup.
- Test whether monitoring can distinguish normal numerical application data
  from a serialized parameter payload.

Keep experiments limited to systems and data you own or are authorized to test.
