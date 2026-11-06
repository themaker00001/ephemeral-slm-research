#!/usr/bin/env python3
"""
Seal (encrypt) and open the packed model payload.

This is an independent implementation of the container format in `src/seal.rs`, built on
the vetted `cryptography` package (ChaCha20-Poly1305, RFC 8439). The committed
`assets/weights.enc` is produced by this script, so the Rust decryptor is checked against
a second implementation rather than only against itself.

Usage:
    python3 scripts/seal_model.py seal assets/weights.bin assets/weights.enc --key-file assets/demo.key
    python3 scripts/seal_model.py open assets/weights.enc out.bin --key-file assets/demo.key [--expect assets/weights.bin]

Container layout (all integers little-endian):
    0   4   magic "ESLS"
    4   4   format version (1)
    8   28  model config: 7 x u32 (version, vocab, ctx, d_model, n_layers, n_heads, d_ff)
    36  8   nonce prefix (random per seal)
    44  4   section count N (= n_layers + 2)
    48  4N  plaintext length of each section
    ..      N sections, each: ciphertext (same length as plaintext) || 16-byte Poly1305 tag

Sections are, in order: embeddings, layer 0 .. layer n-1, final norm + head. Each is
sealed separately so a runtime can decrypt one layer at a time. Section i uses nonce
nonce_prefix || u32(i) and authenticates the whole header plus u32(i) as associated data,
so editing the header, reordering sections, or flipping any ciphertext bit is detected.
"""

import argparse
import os
import struct
import sys

from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

MAGIC = b"ESLS"
FORMAT_VERSION = 1
TAG_LEN = 16
PLAIN_MAGIC = b"ESLM"
PLAIN_HEADER_LEN = 32


def read_key(path):
    text = open(path, "r", encoding="ascii").read().strip()
    key = bytes.fromhex(text)
    if len(key) != 32:
        sys.exit(f"key must be 32 bytes (64 hex chars), got {len(key)} bytes")
    return key


def skip_quantized(buf, off):
    (length,) = struct.unpack_from("<I", buf, off + 4)
    return off + 8 + length


def skip_fp32(buf, off):
    (length,) = struct.unpack_from("<I", buf, off)
    return off + 4 + length


def split_sections(plain):
    """Return (config_words_bytes, [section bytes]) for a plaintext payload."""
    if plain[:4] != PLAIN_MAGIC:
        sys.exit("not a plaintext ESLM payload")
    words = struct.unpack_from("<7I", plain, 4)
    n_layers = words[4]
    off = PLAIN_HEADER_LEN
    bounds = [off]
    off = skip_quantized(plain, off)  # tok_emb
    off = skip_quantized(plain, off)  # pos_emb
    bounds.append(off)
    for _ in range(n_layers):
        off = skip_fp32(plain, off)       # norm1
        for _ in range(4):
            off = skip_quantized(plain, off)  # wq wk wv wo
        off = skip_fp32(plain, off)       # norm2
        off = skip_quantized(plain, off)  # w1
        off = skip_quantized(plain, off)  # w2
        bounds.append(off)
    off = skip_fp32(plain, off)           # final norm
    off = skip_quantized(plain, off)      # head
    bounds.append(off)
    if off != len(plain):
        sys.exit(f"payload has {len(plain) - off} unexpected trailing bytes")
    sections = [plain[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1)]
    return plain[4:PLAIN_HEADER_LEN], sections


def seal(plain, key, nonce_prefix=None):
    config_bytes, sections = split_sections(plain)
    prefix = nonce_prefix if nonce_prefix is not None else os.urandom(8)
    header = bytearray()
    header += MAGIC
    header += struct.pack("<I", FORMAT_VERSION)
    header += config_bytes
    header += prefix
    header += struct.pack("<I", len(sections))
    for s in sections:
        header += struct.pack("<I", len(s))
    aead = ChaCha20Poly1305(key)
    out = bytearray(header)
    for i, s in enumerate(sections):
        nonce = prefix + struct.pack("<I", i)
        aad = bytes(header) + struct.pack("<I", i)
        out += aead.encrypt(nonce, s, aad)
    return bytes(out)


def open_sealed(sealed, key):
    if sealed[:4] != MAGIC:
        sys.exit("not a sealed ESLS payload")
    (version,) = struct.unpack_from("<I", sealed, 4)
    if version != FORMAT_VERSION:
        sys.exit(f"unsupported format version {version}")
    config_bytes = sealed[8:36]
    prefix = sealed[36:44]
    (n,) = struct.unpack_from("<I", sealed, 44)
    lens = struct.unpack_from(f"<{n}I", sealed, 48)
    header_len = 48 + 4 * n
    header = sealed[:header_len]
    aead = ChaCha20Poly1305(key)
    off = header_len
    plain = bytearray(PLAIN_MAGIC + config_bytes)
    for i, length in enumerate(lens):
        chunk = sealed[off:off + length + TAG_LEN]
        off += length + TAG_LEN
        nonce = prefix + struct.pack("<I", i)
        aad = bytes(header) + struct.pack("<I", i)
        plain += aead.decrypt(nonce, chunk, aad)  # raises InvalidTag on tampering
    if off != len(sealed):
        sys.exit("trailing bytes after the last section")
    return bytes(plain)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["seal", "open"])
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--key-file", required=True)
    ap.add_argument("--nonce-hex", help="8-byte nonce prefix as 16 hex chars (tests only; never reuse with a key)")
    ap.add_argument("--expect", help="open: plaintext file the result must equal")
    args = ap.parse_args()

    key = read_key(args.key_file)
    data = open(args.src, "rb").read()
    if args.command == "seal":
        prefix = bytes.fromhex(args.nonce_hex) if args.nonce_hex else None
        if prefix is not None and len(prefix) != 8:
            sys.exit("--nonce-hex must be 8 bytes")
        out = seal(data, key, prefix)
        print(f"[ok] sealed {len(data):,} plaintext bytes -> {len(out):,} bytes in {args.dst}")
    else:
        out = open_sealed(data, key)
        if args.expect and out != open(args.expect, "rb").read():
            sys.exit("[FAIL] opened payload differs from --expect file")
        print(f"[ok] opened {len(data):,} sealed bytes -> {len(out):,} plaintext bytes" + (" (matches --expect)" if args.expect else ""))
    with open(args.dst, "wb") as f:
        f.write(out)


if __name__ == "__main__":
    main()
