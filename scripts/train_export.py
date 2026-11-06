#!/usr/bin/env python3
"""
Ephemeral SLM Research - Model Training and INT8 Binary Weight Exporter.

This script trains a micro-Transformer (LLaMA-style RMSNorm, Multi-Head Attention, SiLU MLP)
on an English text corpus, quantizes the parameters to symmetric INT8 with per-tensor scales,
and serializes them into a packed binary payload for embedding in the Rust runtime.
"""

import os
import struct
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB_SIZE = 128   # Standard ASCII (0..127)
CTX_LEN = 32       # Max context length
D_MODEL = 48       # Hidden dimension
N_LAYERS = 2       # Number of transformer layers
N_HEADS = 3        # Number of attention heads (head_dim = 16)
D_FF = 96          # Feedforward hidden dimension (2 * d_model)

device = "mps" if torch.backends.mps.is_available() else "cpu"

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w

class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = RMSNorm(D_MODEL)
        self.wq = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.wk = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.wv = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.wo = nn.Linear(D_MODEL, D_MODEL, bias=False)

        self.norm2 = RMSNorm(D_MODEL)
        self.w1 = nn.Linear(D_MODEL, D_FF, bias=False)
        self.w2 = nn.Linear(D_FF, D_MODEL, bias=False)

    def forward(self, x):
        h = self.norm1(x)
        B, T, C = h.shape
        q = self.wq(h).view(B, T, N_HEADS, C // N_HEADS).transpose(1, 2)
        k = self.wk(h).view(B, T, N_HEADS, C // N_HEADS).transpose(1, 2)
        v = self.wv(h).view(B, T, N_HEADS, C // N_HEADS).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.wo(attn)

        h2 = self.norm2(x)
        x = x + self.w2(F.silu(self.w1(h2)))
        return x

class EphemeralSLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos_emb = nn.Embedding(CTX_LEN, D_MODEL)
        self.blocks = nn.ModuleList([TransformerBlock() for _ in range(N_LAYERS)])
        self.norm = RMSNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for b in self.blocks:
            x = b(x)
        x = self.norm(x)
        return self.head(x)

def quantize_tensor(tensor: torch.Tensor):
    """Symmetric INT8 quantization: returns (scale: float, quantized: bytes)."""
    t = tensor.detach().cpu().float().numpy().flatten()
    max_val = max(float(abs(t).max()), 1e-8)
    scale = max_val / 127.0
    q = (t / scale).round().clip(-127, 127).astype("int8")
    return scale, q.tobytes()

def train_and_export():
    print(f"[*] Initializing Ephemeral SLM training on device: {device}")
    model = EphemeralSLM().to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"[*] Model parameters: {param_count:,}")

    # Curated knowledge corpus focusing on systems programming, memory safety, and ephemeral SLM concepts
    corpus = """
The ephemeral model lives in transient memory.
Security through memory scrubbing, zeroization, and defensive telemetry.
Small language models operate efficiently on edge devices and enclaves.
Rust guarantees safe systems programming, concurrency, and memory safety.
Ephemeral SLM reconstructs parameters on the fly and scrubs RAM immediately.
The ephemeral intelligence vanishes after generating thoughts.
A micro transformer processes tokens using multi-head attention and feedforward layers.
Safe execution models minimize resident set size and forensic residual exposure.
Transient parameters exist only during active forward passes.
When inference finishes, all owned buffers are wiped and audited.
"""
    clean_corpus = corpus.strip()
    encoded = [ord(c) for c in clean_corpus if ord(c) < VOCAB_SIZE]
    data = torch.tensor(encoded, dtype=torch.long, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
    batch_size = 16
    steps = 450

    print(f"[*] Training for {steps} steps on {len(encoded)} token corpus...")
    model.train()
    for step in range(1, steps + 1):
        ix = torch.randint(0, len(data) - CTX_LEN, (batch_size,))
        x = torch.stack([data[i:i + CTX_LEN] for i in ix])
        y = torch.stack([data[i + 1:i + CTX_LEN + 1] for i in ix])

        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 150 == 0 or step == steps:
            print(f"    Step {step:3d}/{steps} | Loss: {loss.item():.4f}")

    # Verification sample generation
    model.eval()
    test_prompt = "The ephemeral "
    tokens = [ord(c) for c in test_prompt]
    with torch.no_grad():
        for _ in range(35):
            inp = torch.tensor([tokens[-CTX_LEN:]], device=device)
            logits = model(inp)[0, -1]
            next_tok = int(torch.argmax(logits).item())
            tokens.append(next_tok)

    sample_out = "".join(chr(t) for t in tokens)
    print(f"[*] Model verification prompt test:\n    '{sample_out}'")

    # Serialize to INT8 packed binary format
    out_dir = os.path.join(os.path.dirname(__file__), "..", "assets")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "weights.bin")

    payload = bytearray()
    # 1. Magic bytes & Header
    payload.extend(b"ESLM") # Magic: Ephemeral SLM
    payload.extend(struct.pack("<IIIIIII", 1, VOCAB_SIZE, CTX_LEN, D_MODEL, N_LAYERS, N_HEADS, D_FF))

    # Helper to serialize an INT8 quantized tensor
    def pack_quantized(tensor):
        scale, q_bytes = quantize_tensor(tensor)
        payload.extend(struct.pack("<fI", scale, len(q_bytes)))
        payload.extend(q_bytes)

    # Helper to serialize raw FP32 weights (used for small 1D RMSNorm weights)
    def pack_fp32(tensor):
        f_bytes = tensor.detach().cpu().float().numpy().flatten().tobytes()
        payload.extend(struct.pack("<I", len(f_bytes)))
        payload.extend(f_bytes)

    # 2. Global Embeddings
    pack_quantized(model.tok_emb.weight)
    pack_quantized(model.pos_emb.weight)

    # 3. Transformer Layers
    for b in model.blocks:
        pack_fp32(b.norm1.w)
        pack_quantized(b.wq.weight)
        pack_quantized(b.wk.weight)
        pack_quantized(b.wv.weight)
        pack_quantized(b.wo.weight)
        pack_fp32(b.norm2.w)
        pack_quantized(b.w1.weight)
        pack_quantized(b.w2.weight)

    # 4. Final Norm & Head
    pack_fp32(model.norm.w)
    pack_quantized(model.head.weight)

    with open(out_path, "wb") as f:
        f.write(payload)

    print(f"[✓] Serialized packed INT8 model payload to {out_path} ({len(payload):,} bytes)")

if __name__ == "__main__":
    train_and_export()
