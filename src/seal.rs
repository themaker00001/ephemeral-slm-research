//! Sealed (encrypted) model payload: ChaCha20-Poly1305 with one authenticated section
//! per layer, so a runtime can decrypt a single layer at a time.
//!
//! What this is for. Before this module the INT8 weights sat in the executable as plain
//! bytes. Now the executable only carries ciphertext, and the key is supplied at run time
//! (a key file or the `ESLM_KEY` environment variable), so the file alone is not enough to
//! recover the model.
//!
//! What it is not. The cipher below is a from-scratch implementation of RFC 8439 written
//! to keep the crate dependency-free. It is checked against the RFC vectors and against a
//! second implementation (`scripts/seal_model.py`, built on the `cryptography` package),
//! but it has not been audited and is not hardened against side channels beyond avoiding
//! secret-dependent branches and indices. Use a vetted library for anything real. See the
//! README for the threat model: the key lives in process memory while the model runs, so
//! anyone who can read that memory gets the key and each layer as it is decrypted.
//!
//! Container layout (all integers little-endian):
//!
//! ```text
//! 0   4   magic "ESLS"
//! 4   4   format version (1)
//! 8   28  model config: 7 x u32 (version, vocab, ctx, d_model, n_layers, n_heads, d_ff)
//! 36  8   nonce prefix (random per seal)
//! 44  4   section count N (= n_layers + 2)
//! 48  4N  plaintext length of each section
//! ..      N sections, each: ciphertext (same length as plaintext) || 16-byte tag
//! ```
//!
//! Sections are: embeddings, layer 0 .. layer n-1, final norm + head. Section `i` uses the
//! nonce `nonce_prefix || u32(i)` and authenticates the whole header plus `u32(i)` as
//! associated data, so editing the header, reordering sections or flipping any ciphertext
//! bit is detected. One key is used for every section; the nonces differ.

use crate::ephemeral::volatile_zero;
use crate::weights::{ModelConfig, PayloadView};
use std::fmt;

pub const KEY_LEN: usize = 32;
pub const NONCE_LEN: usize = 12;
pub const TAG_LEN: usize = 16;

const MAGIC: &[u8; 4] = b"ESLS";
pub const FORMAT_VERSION: u32 = 1;
const FIXED_HEADER_LEN: usize = 48;
const MAX_SECTIONS: usize = 1024;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SealError {
    Io(String),
    BadKey(String),
    BadMagic,
    UnsupportedVersion(u32),
    Malformed(&'static str),
    /// The authentication tag did not match: wrong key, or the data was modified.
    AuthFailed { section: usize },
}

impl fmt::Display for SealError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SealError::Io(e) => write!(f, "I/O error: {e}"),
            SealError::BadKey(e) => write!(f, "bad key: {e}"),
            SealError::BadMagic => write!(f, "not a sealed model payload (bad magic bytes)"),
            SealError::UnsupportedVersion(v) => write!(f, "unsupported sealed format version {v}"),
            SealError::Malformed(e) => write!(f, "malformed sealed payload: {e}"),
            SealError::AuthFailed { section } => write!(
                f,
                "authentication failed for section {section}: wrong key, or the payload was modified"
            ),
        }
    }
}

impl std::error::Error for SealError {}

// ---------------------------------------------------------------------------
// ChaCha20 (RFC 8439, section 2.1-2.4)
// ---------------------------------------------------------------------------

#[inline(always)]
fn quarter_round(s: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize) {
    s[a] = s[a].wrapping_add(s[b]);
    s[d] ^= s[a];
    s[d] = s[d].rotate_left(16);
    s[c] = s[c].wrapping_add(s[d]);
    s[b] ^= s[c];
    s[b] = s[b].rotate_left(12);
    s[a] = s[a].wrapping_add(s[b]);
    s[d] ^= s[a];
    s[d] = s[d].rotate_left(8);
    s[c] = s[c].wrapping_add(s[d]);
    s[b] ^= s[c];
    s[b] = s[b].rotate_left(7);
}

fn le32(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
}

/// Initial state without the block counter (word 12 is filled in per block).
fn chacha_state(key: &[u8; KEY_LEN], nonce: &[u8; NONCE_LEN]) -> [u32; 16] {
    let mut s = [0u32; 16];
    s[0] = 0x6170_7865;
    s[1] = 0x3320_646e;
    s[2] = 0x7962_2d32;
    s[3] = 0x6b20_6574;
    for i in 0..8 {
        s[4 + i] = le32(key, 4 * i);
    }
    for i in 0..3 {
        s[13 + i] = le32(nonce, 4 * i);
    }
    s
}

/// One 64-byte keystream block for the given counter.
fn chacha_block(state0: &[u32; 16], counter: u32, out: &mut [u8; 64]) {
    let mut init = *state0;
    init[12] = counter;
    let mut s = init;
    for _ in 0..10 {
        quarter_round(&mut s, 0, 4, 8, 12);
        quarter_round(&mut s, 1, 5, 9, 13);
        quarter_round(&mut s, 2, 6, 10, 14);
        quarter_round(&mut s, 3, 7, 11, 15);
        quarter_round(&mut s, 0, 5, 10, 15);
        quarter_round(&mut s, 1, 6, 11, 12);
        quarter_round(&mut s, 2, 7, 8, 13);
        quarter_round(&mut s, 3, 4, 9, 14);
    }
    for i in 0..16 {
        let w = s[i].wrapping_add(init[i]);
        out[4 * i..4 * i + 4].copy_from_slice(&w.to_le_bytes());
    }
    volatile_zero(&mut s);
    volatile_zero(&mut init);
}

/// XOR `data` with the keystream starting at block `counter`.
fn chacha_xor(key: &[u8; KEY_LEN], nonce: &[u8; NONCE_LEN], counter: u32, data: &mut [u8]) {
    assert!(
        (data.len() / 64) < (u32::MAX - counter) as usize,
        "message too long for a 32-bit block counter"
    );
    let mut state0 = chacha_state(key, nonce);
    let mut block = [0u8; 64];
    let mut ctr = counter;
    for chunk in data.chunks_mut(64) {
        chacha_block(&state0, ctr, &mut block);
        for (d, k) in chunk.iter_mut().zip(block.iter()) {
            *d ^= *k;
        }
        ctr = ctr.wrapping_add(1);
    }
    // The keystream XOR ciphertext is plaintext, so do not leave it on the stack.
    volatile_zero(&mut block);
    volatile_zero(&mut state0);
}

// ---------------------------------------------------------------------------
// Poly1305 (RFC 8439, section 2.5), 26-bit limbs
// ---------------------------------------------------------------------------

struct Poly1305 {
    r: [u32; 5],
    h: [u32; 5],
    pad: [u32; 4],
    buf: [u8; 16],
    buf_len: usize,
}

impl Poly1305 {
    fn new(key: &[u8; 32]) -> Self {
        let t0 = le32(key, 0);
        let t1 = le32(key, 4);
        let t2 = le32(key, 8);
        let t3 = le32(key, 12);
        Self {
            r: [
                t0 & 0x03ff_ffff,
                ((t0 >> 26) | (t1 << 6)) & 0x03ff_ff03,
                ((t1 >> 20) | (t2 << 12)) & 0x03ff_c0ff,
                ((t2 >> 14) | (t3 << 18)) & 0x03f0_3fff,
                (t3 >> 8) & 0x000f_ffff,
            ],
            h: [0; 5],
            pad: [le32(key, 16), le32(key, 20), le32(key, 24), le32(key, 28)],
            buf: [0; 16],
            buf_len: 0,
        }
    }

    fn block(&mut self, m: &[u8; 16], hibit: u32) {
        const MASK: u32 = 0x03ff_ffff;
        let t0 = le32(m, 0);
        let t1 = le32(m, 4);
        let t2 = le32(m, 8);
        let t3 = le32(m, 12);

        let h0 = self.h[0].wrapping_add(t0 & MASK);
        let h1 = self.h[1].wrapping_add(((t0 >> 26) | (t1 << 6)) & MASK);
        let h2 = self.h[2].wrapping_add(((t1 >> 20) | (t2 << 12)) & MASK);
        let h3 = self.h[3].wrapping_add(((t2 >> 14) | (t3 << 18)) & MASK);
        let h4 = self.h[4].wrapping_add((t3 >> 8) | hibit);

        let [r0, r1, r2, r3, r4] = self.r.map(u64::from);
        let (s1, s2, s3, s4) = (r1 * 5, r2 * 5, r3 * 5, r4 * 5);
        let [h0, h1, h2, h3, h4] = [h0, h1, h2, h3, h4].map(u64::from);

        let mut d0 = h0 * r0 + h1 * s4 + h2 * s3 + h3 * s2 + h4 * s1;
        let mut d1 = h0 * r1 + h1 * r0 + h2 * s4 + h3 * s3 + h4 * s2;
        let mut d2 = h0 * r2 + h1 * r1 + h2 * r0 + h3 * s4 + h4 * s3;
        let mut d3 = h0 * r3 + h1 * r2 + h2 * r1 + h3 * r0 + h4 * s4;
        let mut d4 = h0 * r4 + h1 * r3 + h2 * r2 + h3 * r1 + h4 * r0;

        let mut c = d0 >> 26;
        let mut n0 = (d0 & MASK as u64) as u32;
        d1 += c;
        c = d1 >> 26;
        let n1 = (d1 & MASK as u64) as u32;
        d2 += c;
        c = d2 >> 26;
        let n2 = (d2 & MASK as u64) as u32;
        d3 += c;
        c = d3 >> 26;
        let n3 = (d3 & MASK as u64) as u32;
        d4 += c;
        c = d4 >> 26;
        let n4 = (d4 & MASK as u64) as u32;
        d0 = n0 as u64 + c * 5;
        c = d0 >> 26;
        n0 = (d0 & MASK as u64) as u32;
        self.h = [n0, n1.wrapping_add(c as u32), n2, n3, n4];
    }

    fn update(&mut self, mut data: &[u8]) {
        if self.buf_len > 0 {
            let take = (16 - self.buf_len).min(data.len());
            self.buf[self.buf_len..self.buf_len + take].copy_from_slice(&data[..take]);
            self.buf_len += take;
            data = &data[take..];
            if self.buf_len < 16 {
                return;
            }
            let b = self.buf;
            self.block(&b, 1 << 24);
            self.buf_len = 0;
        }
        let mut chunks = data.chunks_exact(16);
        for c in &mut chunks {
            let b: [u8; 16] = c.try_into().unwrap();
            self.block(&b, 1 << 24);
        }
        let rest = chunks.remainder();
        self.buf[..rest.len()].copy_from_slice(rest);
        self.buf_len = rest.len();
    }

    fn finalize(mut self) -> [u8; 16] {
        const MASK: u32 = 0x03ff_ffff;
        if self.buf_len > 0 {
            let mut b = [0u8; 16];
            b[..self.buf_len].copy_from_slice(&self.buf[..self.buf_len]);
            b[self.buf_len] = 1;
            self.block(&b, 0);
        }
        let [mut h0, mut h1, mut h2, mut h3, mut h4] = self.h;

        // Fully carry h.
        let mut c = h1 >> 26;
        h1 &= MASK;
        h2 = h2.wrapping_add(c);
        c = h2 >> 26;
        h2 &= MASK;
        h3 = h3.wrapping_add(c);
        c = h3 >> 26;
        h3 &= MASK;
        h4 = h4.wrapping_add(c);
        c = h4 >> 26;
        h4 &= MASK;
        h0 = h0.wrapping_add(c * 5);
        c = h0 >> 26;
        h0 &= MASK;
        h1 = h1.wrapping_add(c);

        // Compute h - p = h + 5 - 2^130.
        let mut g0 = h0.wrapping_add(5);
        c = g0 >> 26;
        g0 &= MASK;
        let mut g1 = h1.wrapping_add(c);
        c = g1 >> 26;
        g1 &= MASK;
        let mut g2 = h2.wrapping_add(c);
        c = g2 >> 26;
        g2 &= MASK;
        let mut g3 = h3.wrapping_add(c);
        c = g3 >> 26;
        g3 &= MASK;
        let g4 = h4.wrapping_add(c).wrapping_sub(1 << 26);

        // Select h if h < p, else h - p, without branching on the value.
        let mut mask = (g4 >> 31).wrapping_sub(1);
        g0 &= mask;
        g1 &= mask;
        g2 &= mask;
        g3 &= mask;
        let g4 = g4 & mask;
        mask = !mask;
        h0 = (h0 & mask) | g0;
        h1 = (h1 & mask) | g1;
        h2 = (h2 & mask) | g2;
        h3 = (h3 & mask) | g3;
        h4 = (h4 & mask) | g4;

        // Pack into 128 bits and add the pad.
        let w0 = (h0 | (h1 << 26)) as u64;
        let w1 = ((h1 >> 6) | (h2 << 20)) as u64;
        let w2 = ((h2 >> 12) | (h3 << 14)) as u64;
        let w3 = ((h3 >> 18) | (h4 << 8)) as u64;

        let mut f = w0 + self.pad[0] as u64;
        let t0 = f as u32;
        f = w1 + self.pad[1] as u64 + (f >> 32);
        let t1 = f as u32;
        f = w2 + self.pad[2] as u64 + (f >> 32);
        let t2 = f as u32;
        f = w3 + self.pad[3] as u64 + (f >> 32);
        let t3 = f as u32;

        let mut tag = [0u8; 16];
        tag[0..4].copy_from_slice(&t0.to_le_bytes());
        tag[4..8].copy_from_slice(&t1.to_le_bytes());
        tag[8..12].copy_from_slice(&t2.to_le_bytes());
        tag[12..16].copy_from_slice(&t3.to_le_bytes());
        volatile_zero(&mut self.r);
        volatile_zero(&mut self.h);
        volatile_zero(&mut self.pad);
        tag
    }
}

// ---------------------------------------------------------------------------
// AEAD construction (RFC 8439, section 2.8)
// ---------------------------------------------------------------------------

fn poly_key(key: &[u8; KEY_LEN], nonce: &[u8; NONCE_LEN]) -> [u8; 32] {
    let state0 = chacha_state(key, nonce);
    let mut block = [0u8; 64];
    chacha_block(&state0, 0, &mut block);
    let mut out = [0u8; 32];
    out.copy_from_slice(&block[..32]);
    volatile_zero(&mut block);
    out
}

fn aead_tag(pk: &[u8; 32], aad: &[&[u8]], ciphertext: &[u8]) -> [u8; TAG_LEN] {
    fn pad16(p: &mut Poly1305, len: usize) {
        let pad = (16 - len % 16) % 16;
        p.update(&[0u8; 16][..pad]);
    }
    let mut p = Poly1305::new(pk);
    let mut aad_len = 0usize;
    for part in aad {
        p.update(part);
        aad_len += part.len();
    }
    pad16(&mut p, aad_len);
    p.update(ciphertext);
    pad16(&mut p, ciphertext.len());
    p.update(&(aad_len as u64).to_le_bytes());
    p.update(&(ciphertext.len() as u64).to_le_bytes());
    p.finalize()
}

/// Constant-time comparison of two tags.
fn tags_equal(a: &[u8; TAG_LEN], b: &[u8]) -> bool {
    if b.len() != TAG_LEN {
        return false;
    }
    let mut diff = 0u8;
    for i in 0..TAG_LEN {
        diff |= a[i] ^ b[i];
    }
    std::hint::black_box(diff) == 0
}

/// Encrypt `plaintext`; returns ciphertext followed by the 16-byte tag.
pub fn aead_encrypt(
    key: &[u8; KEY_LEN],
    nonce: &[u8; NONCE_LEN],
    aad: &[&[u8]],
    plaintext: &[u8],
) -> Vec<u8> {
    let mut pk = poly_key(key, nonce);
    let mut out = Vec::with_capacity(plaintext.len() + TAG_LEN);
    out.extend_from_slice(plaintext);
    chacha_xor(key, nonce, 1, &mut out);
    let tag = aead_tag(&pk, aad, &out);
    volatile_zero(&mut pk);
    out.extend_from_slice(&tag);
    out
}

/// Check the tag of `sealed` (ciphertext || tag) without decrypting anything.
pub fn aead_verify(key: &[u8; KEY_LEN], nonce: &[u8; NONCE_LEN], aad: &[&[u8]], sealed: &[u8]) -> bool {
    if sealed.len() < TAG_LEN {
        return false;
    }
    let (ct, tag) = sealed.split_at(sealed.len() - TAG_LEN);
    let mut pk = poly_key(key, nonce);
    let expected = aead_tag(&pk, aad, ct);
    volatile_zero(&mut pk);
    tags_equal(&expected, tag)
}

/// Verify the tag of `sealed` (ciphertext || tag) and, only if it matches, decrypt into
/// `out`. Unauthenticated plaintext is never produced. `out.len()` must equal the
/// ciphertext length. Returns `false` on any mismatch, leaving `out` untouched.
pub fn aead_decrypt_into(
    key: &[u8; KEY_LEN],
    nonce: &[u8; NONCE_LEN],
    aad: &[&[u8]],
    sealed: &[u8],
    out: &mut [u8],
) -> bool {
    if sealed.len() < TAG_LEN || out.len() != sealed.len() - TAG_LEN {
        return false;
    }
    if !aead_verify(key, nonce, aad, sealed) {
        return false;
    }
    out.copy_from_slice(&sealed[..sealed.len() - TAG_LEN]);
    chacha_xor(key, nonce, 1, out);
    true
}

// ---------------------------------------------------------------------------
// Key handling
// ---------------------------------------------------------------------------

/// A 32-byte key that is volatile-zeroed on drop and never printed.
pub struct SecretKey {
    bytes: [u8; KEY_LEN],
}

impl SecretKey {
    pub fn from_bytes(bytes: [u8; KEY_LEN]) -> Self {
        Self { bytes }
    }

    pub fn as_bytes(&self) -> &[u8; KEY_LEN] {
        &self.bytes
    }

    /// Parse 64 hex characters (surrounding whitespace is ignored).
    pub fn from_hex(text: &[u8]) -> Result<Self, SealError> {
        let t = text.trim_ascii();
        if t.len() != KEY_LEN * 2 {
            return Err(SealError::BadKey(format!(
                "expected {} hex characters, found {}",
                KEY_LEN * 2,
                t.len()
            )));
        }
        let nibble = |c: u8, at: usize| -> Result<u8, SealError> {
            match c {
                b'0'..=b'9' => Ok(c - b'0'),
                b'a'..=b'f' => Ok(c - b'a' + 10),
                b'A'..=b'F' => Ok(c - b'A' + 10),
                _ => Err(SealError::BadKey(format!("non-hex character at position {at}"))),
            }
        };
        let mut bytes = [0u8; KEY_LEN];
        for (i, pair) in t.chunks_exact(2).enumerate() {
            bytes[i] = (nibble(pair[0], 2 * i)? << 4) | nibble(pair[1], 2 * i + 1)?;
        }
        Ok(Self { bytes })
    }

    /// Read a key file containing 64 hex characters.
    pub fn from_file(path: &str) -> Result<Self, SealError> {
        let mut raw = std::fs::read(path).map_err(|e| SealError::Io(format!("{path}: {e}")))?;
        let key = Self::from_hex(&raw);
        volatile_zero(&mut raw);
        key
    }

    /// Read a key from the named environment variable (64 hex characters).
    pub fn from_env(var: &str) -> Result<Self, SealError> {
        let text = std::env::var(var).map_err(|_| SealError::BadKey(format!("{var} is not set")))?;
        let mut raw = text.into_bytes();
        let key = Self::from_hex(&raw);
        volatile_zero(&mut raw);
        key
    }

    /// Use `--key-file <path>` if given, otherwise the `ESLM_KEY` environment variable.
    pub fn from_cli(key_file: Option<&str>) -> Result<Self, SealError> {
        match key_file {
            Some(path) => Self::from_file(path),
            None => Self::from_env("ESLM_KEY").map_err(|_| {
                SealError::BadKey(
                    "no key given: pass --key-file <path> or set ESLM_KEY to 64 hex characters \
                     (the demo key is assets/demo.key)"
                        .to_string(),
                )
            }),
        }
    }

    /// A fresh random key from the operating system.
    pub fn random() -> Result<Self, SealError> {
        use std::io::Read;
        let mut bytes = [0u8; KEY_LEN];
        std::fs::File::open("/dev/urandom")
            .and_then(|mut f| f.read_exact(&mut bytes))
            .map_err(|e| SealError::Io(format!("/dev/urandom: {e}")))?;
        Ok(Self { bytes })
    }

    /// Write the key as hex to a new file (mode 0600 on Unix). Refuses to overwrite.
    pub fn write_to_file(&self, path: &str) -> Result<(), SealError> {
        use std::io::Write;
        let mut opts = std::fs::OpenOptions::new();
        opts.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            opts.mode(0o600);
        }
        let mut f = opts.open(path).map_err(|e| SealError::Io(format!("{path}: {e}")))?;
        let mut line = Vec::with_capacity(KEY_LEN * 2 + 1);
        for b in self.bytes {
            line.push(b"0123456789abcdef"[(b >> 4) as usize]);
            line.push(b"0123456789abcdef"[(b & 15) as usize]);
        }
        line.push(b'\n');
        let r = f.write_all(&line).map_err(|e| SealError::Io(format!("{path}: {e}")));
        volatile_zero(&mut line);
        r
    }
}

impl Drop for SecretKey {
    fn drop(&mut self) {
        volatile_zero(&mut self.bytes);
    }
}

impl fmt::Debug for SecretKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SecretKey(<redacted>)")
    }
}

// ---------------------------------------------------------------------------
// Sealed container
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
struct Section {
    plain_len: usize,
    /// Offset of the ciphertext within the sealed bytes; the tag follows it.
    offset: usize,
}

/// A parsed view over sealed bytes (borrowed; nothing is decrypted yet).
pub struct SealedPayload<'a> {
    raw: &'a [u8],
    pub config: ModelConfig,
    nonce_prefix: [u8; 8],
    header_len: usize,
    sections: Vec<Section>,
}

fn u32_at(raw: &[u8], at: usize) -> u32 {
    le32(raw, at)
}

impl<'a> SealedPayload<'a> {
    pub fn parse(raw: &'a [u8]) -> Result<Self, SealError> {
        if raw.len() < FIXED_HEADER_LEN {
            return Err(SealError::Malformed("shorter than the fixed header"));
        }
        if &raw[0..4] != MAGIC {
            return Err(SealError::BadMagic);
        }
        let version = u32_at(raw, 4);
        if version != FORMAT_VERSION {
            return Err(SealError::UnsupportedVersion(version));
        }
        let config = ModelConfig::from_le_bytes(&raw[8..36]);
        let nonce_prefix: [u8; 8] = raw[36..44].try_into().unwrap();
        let n = u32_at(raw, 44) as usize;
        if !(3..=MAX_SECTIONS).contains(&n) || config.n_layers.checked_add(2) != Some(n) {
            return Err(SealError::Malformed("section count does not match the layer count"));
        }
        let header_len = FIXED_HEADER_LEN + 4 * n;
        if raw.len() < header_len {
            return Err(SealError::Malformed("truncated section table"));
        }
        let mut sections = Vec::with_capacity(n);
        let mut offset = header_len;
        for i in 0..n {
            let plain_len = u32_at(raw, FIXED_HEADER_LEN + 4 * i) as usize;
            let end = offset
                .checked_add(plain_len)
                .and_then(|v| v.checked_add(TAG_LEN))
                .ok_or(SealError::Malformed("section length overflow"))?;
            if end > raw.len() {
                return Err(SealError::Malformed("truncated section data"));
            }
            sections.push(Section { plain_len, offset });
            offset = end;
        }
        if offset != raw.len() {
            return Err(SealError::Malformed("unexpected trailing bytes"));
        }
        Ok(Self {
            raw,
            config,
            nonce_prefix,
            header_len,
            sections,
        })
    }

    pub fn section_count(&self) -> usize {
        self.sections.len()
    }

    /// Plaintext length of section `i`.
    pub fn section_plain_len(&self, i: usize) -> usize {
        self.sections[i].plain_len
    }

    /// Largest plaintext section: the size of the scratch buffer needed to open any one.
    pub fn max_section_len(&self) -> usize {
        self.sections.iter().map(|s| s.plain_len).max().unwrap_or(0)
    }

    /// Size of the sealed bytes (what is embedded in the executable).
    pub fn total_bytes(&self) -> usize {
        self.raw.len()
    }

    fn nonce(&self, i: usize) -> [u8; NONCE_LEN] {
        let mut n = [0u8; NONCE_LEN];
        n[..8].copy_from_slice(&self.nonce_prefix);
        n[8..].copy_from_slice(&(i as u32).to_le_bytes());
        n
    }

    /// Check the authentication tag of section `i` without decrypting it. Fails for a wrong
    /// key or any modification of the header or the section.
    pub fn verify_section(&self, i: usize, key: &SecretKey) -> Result<(), SealError> {
        let s = self
            .sections
            .get(i)
            .ok_or(SealError::Malformed("section index out of range"))?;
        let idx = (i as u32).to_le_bytes();
        let aad: [&[u8]; 2] = [&self.raw[..self.header_len], &idx];
        let sealed = &self.raw[s.offset..s.offset + s.plain_len + TAG_LEN];
        if aead_verify(key.as_bytes(), &self.nonce(i), &aad, sealed) {
            Ok(())
        } else {
            Err(SealError::AuthFailed { section: i })
        }
    }

    /// Check every section's tag, without producing any plaintext. Cheap, and it lets a
    /// caller reject a wrong key or a damaged file before doing any work.
    pub fn verify_all(&self, key: &SecretKey) -> Result<(), SealError> {
        (0..self.sections.len()).try_for_each(|i| self.verify_section(i, key))
    }

    /// Authenticate and decrypt section `i` into `out` (exactly the section's plaintext
    /// length). On failure nothing is written to `out`.
    pub fn open_section(&self, i: usize, key: &SecretKey, out: &mut [u8]) -> Result<(), SealError> {
        let s = self
            .sections
            .get(i)
            .ok_or(SealError::Malformed("section index out of range"))?;
        if out.len() != s.plain_len {
            return Err(SealError::Malformed("output buffer has the wrong length"));
        }
        let idx = (i as u32).to_le_bytes();
        let aad: [&[u8]; 2] = [&self.raw[..self.header_len], &idx];
        let sealed = &self.raw[s.offset..s.offset + s.plain_len + TAG_LEN];
        if aead_decrypt_into(key.as_bytes(), &self.nonce(i), &aad, sealed, out) {
            Ok(())
        } else {
            Err(SealError::AuthFailed { section: i })
        }
    }

    /// Rebuild the original plaintext payload (the `ESLM` layout). The result holds the
    /// whole model in the clear: use it for tests and tooling, not for inference.
    pub fn open_all(&self, key: &SecretKey) -> Result<Vec<u8>, SealError> {
        let mut out = Vec::new();
        out.extend_from_slice(b"ESLM");
        out.extend_from_slice(&self.raw[8..36]);
        for i in 0..self.sections.len() {
            let start = out.len();
            out.resize(start + self.sections[i].plain_len, 0);
            self.open_section(i, key, &mut out[start..])?;
        }
        Ok(out)
    }
}

/// Seal a plaintext `ESLM` payload. `nonce_prefix` must be unique per (key, payload): use
/// [`seal_random_nonce`] unless you are writing a test.
pub fn seal_payload(plain: &[u8], key: &SecretKey, nonce_prefix: [u8; 8]) -> Result<Vec<u8>, SealError> {
    let view = PayloadView::parse(plain)
        .map_err(|_| SealError::Malformed("plaintext payload does not parse"))?;
    let ranges = view.section_ranges();

    let mut header = Vec::with_capacity(FIXED_HEADER_LEN + 4 * ranges.len());
    header.extend_from_slice(MAGIC);
    header.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    header.extend_from_slice(&plain[4..32]);
    header.extend_from_slice(&nonce_prefix);
    header.extend_from_slice(&(ranges.len() as u32).to_le_bytes());
    for (a, b) in &ranges {
        header.extend_from_slice(&((b - a) as u32).to_le_bytes());
    }

    let mut out = header.clone();
    for (i, (a, b)) in ranges.iter().enumerate() {
        let mut nonce = [0u8; NONCE_LEN];
        nonce[..8].copy_from_slice(&nonce_prefix);
        nonce[8..].copy_from_slice(&(i as u32).to_le_bytes());
        let idx = (i as u32).to_le_bytes();
        let aad: [&[u8]; 2] = [&header, &idx];
        out.extend_from_slice(&aead_encrypt(key.as_bytes(), &nonce, &aad, &plain[*a..*b]));
    }
    Ok(out)
}

/// [`seal_payload`] with a fresh random nonce prefix from the operating system.
pub fn seal_random_nonce(plain: &[u8], key: &SecretKey) -> Result<Vec<u8>, SealError> {
    use std::io::Read;
    let mut prefix = [0u8; 8];
    std::fs::File::open("/dev/urandom")
        .and_then(|mut f| f.read_exact(&mut prefix))
        .map_err(|e| SealError::Io(format!("/dev/urandom: {e}")))?;
    seal_payload(plain, key, prefix)
}

#[cfg(test)]
mod tests {
    use super::*;

    const PLAIN: &[u8] = include_bytes!("../assets/weights.bin");

    fn unhex(s: &str) -> Vec<u8> {
        (0..s.len() / 2)
            .map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap())
            .collect()
    }

    fn arr<const N: usize>(v: &[u8]) -> [u8; N] {
        v.try_into().unwrap()
    }

    // Vectors below come from RFC 8439 and were confirmed with Python's `cryptography`.

    #[test]
    fn chacha20_block_matches_rfc8439() {
        let key: [u8; 32] = std::array::from_fn(|i| i as u8);
        let nonce: [u8; 12] = arr(&unhex("000000090000004a00000000"));
        let state = chacha_state(&key, &nonce);
        let mut out = [0u8; 64];
        chacha_block(&state, 1, &mut out);
        assert_eq!(
            out.to_vec(),
            unhex(
                "10f1e7e4d13b5915500fdd1fa32071c4c7d1f4c733c068030422aa9ac3d46c4e\
                 d2826446079faa0914c2d705d98b02a2b5129cd1de164eb9cbd083e8a2503c4e"
            )
        );
    }

    #[test]
    fn poly1305_matches_rfc8439() {
        let key: [u8; 32] =
            arr(&unhex("85d6be7857556d337f4452fe42d506a80103808afb0db2fd4abff6af4149f51b"));
        let msg = b"Cryptographic Forum Research Group";
        let mut p = Poly1305::new(&key);
        p.update(msg);
        assert_eq!(p.finalize().to_vec(), unhex("a8061dc1305136c6c22b8baf0c0127a9"));
    }

    #[test]
    fn poly1305_streaming_equals_one_shot() {
        let key: [u8; 32] = std::array::from_fn(|i| (i * 13 + 1) as u8);
        let msg: Vec<u8> = (0..300u32).map(|i| (i * 7 + 3) as u8).collect();
        let mut whole = Poly1305::new(&key);
        whole.update(&msg);
        let expected = whole.finalize();
        for split in 0..=msg.len() {
            let mut p = Poly1305::new(&key);
            p.update(&msg[..split]);
            p.update(&msg[split..]);
            assert_eq!(p.finalize(), expected, "split at {split}");
        }
    }

    #[test]
    fn aead_matches_rfc8439_section_2_8_2() {
        let key: [u8; 32] = std::array::from_fn(|i| 0x80 + i as u8);
        let nonce: [u8; 12] = arr(&unhex("070000004041424344454647"));
        let aad = unhex("50515253c0c1c2c3c4c5c6c7");
        let pt = b"Ladies and Gentlemen of the class of '99: If I could offer you only one tip for the future, sunscreen would be it.";
        let sealed = aead_encrypt(&key, &nonce, &[&aad], pt);
        assert_eq!(&sealed[..16], &unhex("d31a8d34648e60db7b86afbc53ef7ec2")[..]);
        assert_eq!(&sealed[sealed.len() - 16..], &unhex("1ae10b594f09e26a7e902ecbd0600691")[..]);

        let mut back = vec![0u8; pt.len()];
        assert!(aead_decrypt_into(&key, &nonce, &[&aad], &sealed, &mut back));
        assert_eq!(&back[..], &pt[..]);
    }

    #[test]
    fn aead_matches_python_for_many_lengths() {
        let key: [u8; 32] = std::array::from_fn(|i| (i * 7 + 3) as u8);
        let nonce: [u8; 12] = std::array::from_fn(|i| (i * 11 + 5) as u8);
        let expected = [
            (0, "8dda000cf4057e4b5850f2226b360ac1"),
            (1, "da6e0d0cda7a0f650b2c612bf1a72cfa"),
            (15, "946d3133f7f73d8fd604b80b8d003b02"),
            (16, "1cd4c6f936ba3e21aa7703c912ef8d25"),
            (17, "57d03f332d780fab1a2bb99427cb1f1b"),
            (63, "50495cb084db6408cb13bac23788b67c"),
            (64, "0f835779fea9aa674dedac301ca54c92"),
            (65, "cf3f6c873572f558c1c91bc21c6618ac"),
            (127, "888fc69f6549d3e8c4dfab7b714937e1"),
            (128, "9ec3e08dc25e92393816bf0ea506b572"),
            (129, "0691ec77949df8155c2ee121325bec51"),
            (1000, "43ade245f7420997994e1fbbdcb9d643"),
            (4096, "6a0b1779c32836a856629f97ce9722d7"),
        ];
        for (len, tag) in expected {
            let pt: Vec<u8> = (0..len).map(|i| ((i * 31 + 7 + (i >> 8) * 3) & 0xff) as u8).collect();
            let aad: Vec<u8> = (0..len % 13).map(|i| ((i * 5 + 1) & 0xff) as u8).collect();
            let sealed = aead_encrypt(&key, &nonce, &[&aad], &pt);
            assert_eq!(&sealed[len..], &unhex(tag)[..], "tag mismatch at length {len}");
            let mut back = vec![0u8; len];
            assert!(aead_decrypt_into(&key, &nonce, &[&aad], &sealed, &mut back));
            assert_eq!(back, pt, "round trip at length {len}");
        }
    }

    #[test]
    fn aead_rejects_any_modification() {
        let key = [7u8; 32];
        let nonce = [9u8; 12];
        let aad = b"header";
        let pt: Vec<u8> = (0..200u32).map(|i| i as u8).collect();
        let sealed = aead_encrypt(&key, &nonce, &[aad], &pt);
        let mut out = vec![0xAAu8; pt.len()];

        assert!(aead_decrypt_into(&key, &nonce, &[aad], &sealed, &mut out));

        for pos in 0..sealed.len() {
            let mut bad = sealed.clone();
            bad[pos] ^= 0x01;
            let mut sink = vec![0xAAu8; pt.len()];
            assert!(!aead_decrypt_into(&key, &nonce, &[aad], &bad, &mut sink), "byte {pos}");
            assert!(sink.iter().all(|&b| b == 0xAA), "plaintext leaked on failure at byte {pos}");
        }
        let mut sink = vec![0u8; pt.len()];
        assert!(!aead_decrypt_into(&[8u8; 32], &nonce, &[aad], &sealed, &mut sink), "wrong key");
        assert!(!aead_decrypt_into(&key, &[1u8; 12], &[aad], &sealed, &mut sink), "wrong nonce");
        assert!(!aead_decrypt_into(&key, &nonce, &[b"other"], &sealed, &mut sink), "wrong aad");
        assert!(!aead_decrypt_into(&key, &nonce, &[aad], &sealed[..sealed.len() - 1], &mut sink), "truncated");
    }

    fn test_key() -> SecretKey {
        SecretKey::from_bytes(std::array::from_fn(|i| (i * 5 + 17) as u8))
    }

    #[test]
    fn container_round_trips_the_real_model() {
        let key = test_key();
        let sealed = seal_payload(PLAIN, &key, *b"testpref").unwrap();
        let view = SealedPayload::parse(&sealed).unwrap();
        assert_eq!(view.section_count(), view.config.n_layers + 2);
        assert_eq!(view.open_all(&key).unwrap(), PLAIN);
        // No plaintext tensor window survives in the sealed bytes.
        assert!(!sealed.windows(32).any(|w| PLAIN.windows(32).step_by(499).any(|p| p == w)));
    }

    #[test]
    fn container_rejects_wrong_key_and_tampering() {
        let key = test_key();
        let sealed = seal_payload(PLAIN, &key, *b"testpref").unwrap();
        let view = SealedPayload::parse(&sealed).unwrap();

        let other = SecretKey::from_bytes([1u8; 32]);
        let mut buf = vec![0u8; view.section_plain_len(1)];
        assert_eq!(view.open_section(1, &other, &mut buf), Err(SealError::AuthFailed { section: 1 }));
        assert!(buf.iter().all(|&b| b == 0), "nothing may be written on failure");

        // Flip one byte in every region: header fields, section table, ciphertext, tag.
        for pos in [5, 12, 40, 46, 50, 70, sealed.len() / 2, sealed.len() - 1] {
            let mut bad = sealed.clone();
            bad[pos] ^= 0x80;
            let ok = SealedPayload::parse(&bad)
                .and_then(|v| v.open_all(&key).map(|_| ()))
                .is_ok();
            assert!(!ok, "tampering at byte {pos} went undetected");
        }
    }

    #[test]
    fn verify_checks_tags_without_decrypting() {
        let key = test_key();
        let sealed = seal_payload(PLAIN, &key, *b"testpref").unwrap();
        let view = SealedPayload::parse(&sealed).unwrap();
        assert_eq!(view.verify_all(&key), Ok(()));
        assert_eq!(
            view.verify_all(&SecretKey::from_bytes([1u8; 32])),
            Err(SealError::AuthFailed { section: 0 })
        );
        // Damage inside the last section is reported against that section.
        let mut bad = sealed.clone();
        let last = bad.len() - 20;
        bad[last] ^= 1;
        let v = SealedPayload::parse(&bad).unwrap();
        assert_eq!(v.verify_all(&key), Err(SealError::AuthFailed { section: v.section_count() - 1 }));
        assert_eq!(v.verify_section(0, &key), Ok(()));
        assert!(matches!(v.verify_section(99, &key), Err(SealError::Malformed(_))));
    }

    #[test]
    fn container_detects_swapped_sections() {
        let key = test_key();
        let sealed = seal_payload(PLAIN, &key, *b"testpref").unwrap();
        let view = SealedPayload::parse(&sealed).unwrap();
        // Layers 0 and 1 have the same length, so their sealed bytes can be exchanged.
        assert_eq!(view.section_plain_len(1), view.section_plain_len(2));
        let len = view.section_plain_len(1) + TAG_LEN;
        let a = view.sections[1].offset;
        let b = view.sections[2].offset;
        let mut bad = sealed.clone();
        bad[a..a + len].copy_from_slice(&sealed[b..b + len]);
        bad[b..b + len].copy_from_slice(&sealed[a..a + len]);
        let v = SealedPayload::parse(&bad).unwrap();
        assert!(v.open_all(&key).is_err(), "swapped sections must not authenticate");
    }

    #[test]
    fn container_parse_rejects_malformed_input() {
        let key = test_key();
        let sealed = seal_payload(PLAIN, &key, *b"testpref").unwrap();
        assert!(matches!(SealedPayload::parse(&sealed[..10]), Err(SealError::Malformed(_))));
        assert!(matches!(SealedPayload::parse(&sealed[..sealed.len() - 1]), Err(SealError::Malformed(_))));
        let mut extra = sealed.clone();
        extra.push(0);
        assert!(matches!(SealedPayload::parse(&extra), Err(SealError::Malformed(_))));
        let mut bad_magic = sealed.clone();
        bad_magic[0] = b'X';
        assert!(matches!(SealedPayload::parse(&bad_magic), Err(SealError::BadMagic)));
        let mut bad_ver = sealed.clone();
        bad_ver[4] = 2;
        assert!(matches!(SealedPayload::parse(&bad_ver), Err(SealError::UnsupportedVersion(2))));
        assert!(matches!(SealedPayload::parse(PLAIN), Err(SealError::BadMagic)));
    }

    #[test]
    fn key_parsing_and_redaction() {
        let hex = "00112233445566778899aabbccddeeff00112233445566778899AABBCCDDEEFF";
        let key = SecretKey::from_hex(format!("  {hex}\n").as_bytes()).unwrap();
        assert_eq!(key.as_bytes()[0], 0x00);
        assert_eq!(key.as_bytes()[15], 0xff);
        assert_eq!(key.as_bytes()[31], 0xff);
        assert_eq!(format!("{key:?}"), "SecretKey(<redacted>)");

        let too_short = SecretKey::from_hex(b"abcd").unwrap_err();
        assert!(matches!(too_short, SealError::BadKey(_)));
        let bad = format!("zz{}", &hex[2..]);
        let err = SecretKey::from_hex(bad.as_bytes()).unwrap_err().to_string();
        assert!(!err.contains(&hex[2..10]), "error text must not echo key material: {err}");
    }
}
