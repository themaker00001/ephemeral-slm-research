//! Model payload: INT8 quantization decoding and the sealed (encrypted) payload that is
//! embedded in the executable.
//!
//! Parameters are stored as symmetric INT8 arrays with per-tensor FP32 scales. The
//! executable embeds the model only in **sealed** form (`assets/weights.enc`, see
//! [`crate::seal`]): ChaCha20-Poly1305 ciphertext with one section per layer. The plaintext
//! layout parsed here ([`PayloadView`] and the `parse_*_section` functions) is what a
//! section contains after it has been decrypted into a scratch buffer.

use std::convert::TryInto;

/// The model as shipped in the executable: encrypted and authenticated, one section per
/// layer. It is useless without the key, which is never embedded.
pub static EMBEDDED_SEALED_PAYLOAD: &[u8] = include_bytes!("../assets/weights.enc");

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelConfig {
    pub version: u32,
    pub vocab_size: usize,
    pub ctx_len: usize,
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub d_ff: usize,
}

impl ModelConfig {
    /// Decode the 7 little-endian `u32` config words (version, vocab, ctx, d_model,
    /// n_layers, n_heads, d_ff). `bytes` must hold at least 28 bytes.
    pub fn from_le_bytes(bytes: &[u8]) -> Self {
        let word = |i: usize| u32::from_le_bytes(bytes[4 * i..4 * i + 4].try_into().unwrap());
        Self {
            version: word(0),
            vocab_size: word(1) as usize,
            ctx_len: word(2) as usize,
            d_model: word(3) as usize,
            n_layers: word(4) as usize,
            n_heads: word(5) as usize,
            d_ff: word(6) as usize,
        }
    }
}

/// A slice referencing quantized INT8 data inside a payload or a decrypted section.
#[derive(Debug, Clone, Copy)]
pub struct QuantizedSlice<'a> {
    pub scale: f32,
    pub data: &'a [i8],
    pub shape: &'a [usize],
}

impl<'a> QuantizedSlice<'a> {
    /// Dequantize stored INT8 values into an allocated FP32 destination buffer.
    #[inline]
    pub fn dequantize_into(&self, dest: &mut [f32]) {
        assert_eq!(
            dest.len(),
            self.data.len(),
            "Destination buffer size mismatch: expected {}, got {}",
            self.data.len(),
            dest.len()
        );
        let scale = self.scale;
        for (d, &q) in dest.iter_mut().zip(self.data.iter()) {
            *d = q as f32 * scale;
        }
    }

    pub fn num_elements(&self) -> usize {
        self.data.len()
    }
}

/// Raw little-endian FP32 weights (used for small 1D normalization vectors where
/// quantization offers minimal gain).
///
/// The bytes are kept as `&[u8]` and decoded with `f32::from_le_bytes`. A payload only
/// guarantees 1-byte alignment, so reinterpreting it as `&[f32]` would be undefined
/// behaviour whenever an offset is not 4-byte aligned.
#[derive(Debug, Clone, Copy)]
pub struct Fp32Slice<'a> {
    bytes: &'a [u8],
}

impl<'a> Fp32Slice<'a> {
    /// Number of `f32` values in the slice.
    #[inline]
    pub fn len(&self) -> usize {
        self.bytes.len() / std::mem::size_of::<f32>()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    /// Decode the stored values into `dest`.
    #[inline]
    pub fn copy_into(&self, dest: &mut [f32]) {
        assert_eq!(dest.len(), self.len());
        for (d, chunk) in dest.iter_mut().zip(self.bytes.chunks_exact(4)) {
            *d = f32::from_le_bytes(chunk.try_into().unwrap());
        }
    }
}

pub struct LayerSlices<'a> {
    pub norm1: Fp32Slice<'a>,
    pub wq: QuantizedSlice<'a>,
    pub wk: QuantizedSlice<'a>,
    pub wv: QuantizedSlice<'a>,
    pub wo: QuantizedSlice<'a>,
    pub norm2: Fp32Slice<'a>,
    pub w1: QuantizedSlice<'a>,
    pub w2: QuantizedSlice<'a>,
}

// ---------------------------------------------------------------------------
// Plaintext section parsing (works on any byte buffer, e.g. a decrypted section)
// ---------------------------------------------------------------------------

static MAT_SHAPE: [usize; 2] = [48, 48];
static W1_SHAPE: [usize; 2] = [96, 48];
static W2_SHAPE: [usize; 2] = [48, 96];
static TOK_SHAPE: [usize; 2] = [128, 48];
static POS_SHAPE: [usize; 2] = [32, 48];
static HEAD_SHAPE: [usize; 2] = [128, 48];

fn take_quantized<'a>(buf: &'a [u8], offset: &mut usize, shape: &'static [usize]) -> QuantizedSlice<'a> {
    let scale = f32::from_le_bytes(buf[*offset..*offset + 4].try_into().unwrap());
    let length = u32::from_le_bytes(buf[*offset + 4..*offset + 8].try_into().unwrap()) as usize;
    *offset += 8;
    let byte_slice = &buf[*offset..*offset + length];
    *offset += length;
    // SAFETY: `i8` and `u8` have identical size and alignment (1), and every bit
    // pattern is a valid `i8`, so this reinterpretation is sound for any offset.
    let data = unsafe { std::slice::from_raw_parts(byte_slice.as_ptr() as *const i8, length) };
    QuantizedSlice { scale, data, shape }
}

fn take_fp32<'a>(buf: &'a [u8], offset: &mut usize) -> Fp32Slice<'a> {
    let byte_len = u32::from_le_bytes(buf[*offset..*offset + 4].try_into().unwrap()) as usize;
    *offset += 4;
    let bytes = &buf[*offset..*offset + byte_len];
    *offset += byte_len;
    Fp32Slice { bytes }
}

/// Parse the embeddings section: token embedding then position embedding.
pub fn parse_embedding_section(buf: &[u8]) -> (QuantizedSlice<'_>, QuantizedSlice<'_>) {
    let mut cur = 0;
    let tok = take_quantized(buf, &mut cur, &TOK_SHAPE);
    let pos = take_quantized(buf, &mut cur, &POS_SHAPE);
    (tok, pos)
}

/// Parse one transformer layer section.
pub fn parse_layer_section(buf: &[u8]) -> LayerSlices<'_> {
    let mut cur = 0;
    let norm1 = take_fp32(buf, &mut cur);
    let wq = take_quantized(buf, &mut cur, &MAT_SHAPE);
    let wk = take_quantized(buf, &mut cur, &MAT_SHAPE);
    let wv = take_quantized(buf, &mut cur, &MAT_SHAPE);
    let wo = take_quantized(buf, &mut cur, &MAT_SHAPE);
    let norm2 = take_fp32(buf, &mut cur);
    let w1 = take_quantized(buf, &mut cur, &W1_SHAPE);
    let w2 = take_quantized(buf, &mut cur, &W2_SHAPE);
    LayerSlices {
        norm1,
        wq,
        wk,
        wv,
        wo,
        norm2,
        w1,
        w2,
    }
}

/// Parse the final section: output norm then the head.
pub fn parse_final_section(buf: &[u8]) -> (Fp32Slice<'_>, QuantizedSlice<'_>) {
    let mut cur = 0;
    let norm = take_fp32(buf, &mut cur);
    let head = take_quantized(buf, &mut cur, &HEAD_SHAPE);
    (norm, head)
}

/// View over a **plaintext** payload (the `ESLM` layout produced by
/// `scripts/train_export.py`). The executable no longer embeds one; this is used by the
/// sealing tool, by tests, and to interpret decrypted sections.
pub struct PayloadView<'a> {
    raw: &'a [u8],
    pub config: ModelConfig,
    embedding_offset: usize,
    layer_offsets: Vec<usize>,
    final_offset: usize,
}

impl<'a> PayloadView<'a> {
    pub fn parse(raw: &'a [u8]) -> Result<Self, String> {
        if raw.len() < 32 {
            return Err("Model payload too short for header".into());
        }
        if &raw[0..4] != b"ESLM" {
            return Err(format!("Invalid magic bytes: {:?}", &raw[0..4]));
        }

        let config = ModelConfig::from_le_bytes(&raw[4..32]);
        let n_layers = config.n_layers;

        let embedding_offset = 32;
        let mut offset = embedding_offset;
        // Skip embedding tensors to locate layer offsets
        Self::skip_quantized(raw, &mut offset)?; // tok_emb
        Self::skip_quantized(raw, &mut offset)?; // pos_emb

        let mut layer_offsets = Vec::with_capacity(n_layers.min(1024));
        for _ in 0..n_layers {
            layer_offsets.push(offset);
            Self::skip_fp32(raw, &mut offset)?;      // norm1
            Self::skip_quantized(raw, &mut offset)?; // wq
            Self::skip_quantized(raw, &mut offset)?; // wk
            Self::skip_quantized(raw, &mut offset)?; // wv
            Self::skip_quantized(raw, &mut offset)?; // wo
            Self::skip_fp32(raw, &mut offset)?;      // norm2
            Self::skip_quantized(raw, &mut offset)?; // w1
            Self::skip_quantized(raw, &mut offset)?; // w2
        }

        let final_offset = offset;
        Self::skip_fp32(raw, &mut offset)?;      // final norm
        Self::skip_quantized(raw, &mut offset)?; // head
        if offset != raw.len() {
            return Err(format!("{} unexpected trailing bytes", raw.len() - offset));
        }

        Ok(Self {
            raw,
            config,
            embedding_offset,
            layer_offsets,
            final_offset,
        })
    }

    fn skip_quantized(raw: &[u8], offset: &mut usize) -> Result<(), String> {
        if *offset + 8 > raw.len() {
            return Err("Unexpected EOF parsing quantized header".into());
        }
        let length = u32::from_le_bytes(raw[*offset + 4..*offset + 8].try_into().unwrap()) as usize;
        *offset += 8 + length;
        if *offset > raw.len() {
            return Err("Unexpected EOF parsing quantized data".into());
        }
        Ok(())
    }

    fn skip_fp32(raw: &[u8], offset: &mut usize) -> Result<(), String> {
        if *offset + 4 > raw.len() {
            return Err("Unexpected EOF parsing fp32 header".into());
        }
        let byte_len = u32::from_le_bytes(raw[*offset..*offset + 4].try_into().unwrap()) as usize;
        *offset += 4 + byte_len;
        if *offset > raw.len() {
            return Err("Unexpected EOF parsing fp32 data".into());
        }
        Ok(())
    }

    pub fn get_embeddings(&self) -> (QuantizedSlice<'a>, QuantizedSlice<'a>) {
        parse_embedding_section(&self.raw[self.embedding_offset..])
    }

    pub fn get_layer_slices(&self, layer_idx: usize) -> LayerSlices<'a> {
        parse_layer_section(&self.raw[self.layer_offsets[layer_idx]..])
    }

    pub fn get_final_slices(&self) -> (Fp32Slice<'a>, QuantizedSlice<'a>) {
        parse_final_section(&self.raw[self.final_offset..])
    }

    pub fn total_payload_bytes(&self) -> usize {
        self.raw.len()
    }

    /// Byte ranges of the independently sealable sections, in order: embeddings, each
    /// layer, then the final norm and head. Together they cover everything after the
    /// 32-byte header.
    pub fn section_ranges(&self) -> Vec<(usize, usize)> {
        let mut starts = vec![self.embedding_offset];
        starts.extend(self.layer_offsets.iter().copied());
        starts.push(self.final_offset);
        let mut ranges = Vec::with_capacity(starts.len());
        for (i, &start) in starts.iter().enumerate() {
            let end = starts.get(i + 1).copied().unwrap_or(self.raw.len());
            ranges.push((start, end));
        }
        ranges
    }
}
