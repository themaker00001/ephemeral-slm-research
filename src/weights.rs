//! Source-embedded model payload and INT8 quantization decoding.
//!
//! Models parameters are stored as symmetric INT8 quantized arrays with per-tensor
//! FP32 scales. The weights binary is embedded directly into the executable's
//! `.rodata` section at compile time via `include_bytes!`.

use std::convert::TryInto;

pub static EMBEDDED_MODEL_PAYLOAD: &[u8] = include_bytes!("../assets/weights.bin");

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

/// A slice referencing quantized INT8 data directly from the read-only static binary payload.
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

/// Raw FP32 weights (used for small 1D normalization vectors where quantization offers minimal gain).
#[derive(Debug, Clone, Copy)]
pub struct Fp32Slice<'a> {
    pub data: &'a [f32],
}

impl<'a> Fp32Slice<'a> {
    #[inline]
    pub fn copy_into(&self, dest: &mut [f32]) {
        assert_eq!(dest.len(), self.data.len());
        dest.copy_from_slice(self.data);
    }
}

/// Lightweight, zero-copy view over the embedded model payload.
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

        let mut offset = 4;
        let read_u32 = |cur: &mut usize| -> u32 {
            let val = u32::from_le_bytes(raw[*cur..*cur + 4].try_into().unwrap());
            *cur += 4;
            val
        };

        let version = read_u32(&mut offset);
        let vocab_size = read_u32(&mut offset) as usize;
        let ctx_len = read_u32(&mut offset) as usize;
        let d_model = read_u32(&mut offset) as usize;
        let n_layers = read_u32(&mut offset) as usize;
        let n_heads = read_u32(&mut offset) as usize;
        let d_ff = read_u32(&mut offset) as usize;

        let config = ModelConfig {
            version,
            vocab_size,
            ctx_len,
            d_model,
            n_layers,
            n_heads,
            d_ff,
        };

        let embedding_offset = offset;
        // Skip embedding tensors to locate layer offsets
        Self::skip_quantized(raw, &mut offset)?; // tok_emb
        Self::skip_quantized(raw, &mut offset)?; // pos_emb

        let mut layer_offsets = Vec::with_capacity(n_layers);
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

    fn read_quantized_at(&self, offset: &mut usize, shape: &'static [usize]) -> QuantizedSlice<'a> {
        let scale = f32::from_le_bytes(self.raw[*offset..*offset + 4].try_into().unwrap());
        let length = u32::from_le_bytes(self.raw[*offset + 4..*offset + 8].try_into().unwrap()) as usize;
        *offset += 8;
        let byte_slice = &self.raw[*offset..*offset + length];
        *offset += length;
        let data = unsafe {
            std::slice::from_raw_parts(byte_slice.as_ptr() as *const i8, length)
        };
        QuantizedSlice { scale, data, shape }
    }

    fn read_fp32_at(&self, offset: &mut usize, _shape: &'static [usize]) -> Fp32Slice<'a> {
        let byte_len = u32::from_le_bytes(self.raw[*offset..*offset + 4].try_into().unwrap()) as usize;
        *offset += 4;
        let byte_slice = &self.raw[*offset..*offset + byte_len];
        *offset += byte_len;
        let float_len = byte_len / std::mem::size_of::<f32>();
        let data = unsafe {
            std::slice::from_raw_parts(byte_slice.as_ptr() as *const f32, float_len)
        };
        Fp32Slice { data }
    }

    pub fn get_embeddings(&self) -> (QuantizedSlice<'a>, QuantizedSlice<'a>) {
        let mut cur = self.embedding_offset;
        static TOK_SHAPE: [usize; 2] = [128, 48];
        static POS_SHAPE: [usize; 2] = [32, 48];
        let tok = self.read_quantized_at(&mut cur, &TOK_SHAPE);
        let pos = self.read_quantized_at(&mut cur, &POS_SHAPE);
        (tok, pos)
    }

    pub fn get_layer_slices(&self, layer_idx: usize) -> LayerSlices<'a> {
        let mut cur = self.layer_offsets[layer_idx];
        static NORM_SHAPE: [usize; 1] = [48];
        static MAT_SHAPE: [usize; 2] = [48, 48];
        static W1_SHAPE: [usize; 2] = [96, 48];
        static W2_SHAPE: [usize; 2] = [48, 96];

        let norm1 = self.read_fp32_at(&mut cur, &NORM_SHAPE);
        let wq = self.read_quantized_at(&mut cur, &MAT_SHAPE);
        let wk = self.read_quantized_at(&mut cur, &MAT_SHAPE);
        let wv = self.read_quantized_at(&mut cur, &MAT_SHAPE);
        let wo = self.read_quantized_at(&mut cur, &MAT_SHAPE);
        let norm2 = self.read_fp32_at(&mut cur, &NORM_SHAPE);
        let w1 = self.read_quantized_at(&mut cur, &W1_SHAPE);
        let w2 = self.read_quantized_at(&mut cur, &W2_SHAPE);

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

    pub fn get_final_slices(&self) -> (Fp32Slice<'a>, QuantizedSlice<'a>) {
        let mut cur = self.final_offset;
        static NORM_SHAPE: [usize; 1] = [48];
        static HEAD_SHAPE: [usize; 2] = [128, 48];
        let norm = self.read_fp32_at(&mut cur, &NORM_SHAPE);
        let head = self.read_quantized_at(&mut cur, &HEAD_SHAPE);
        (norm, head)
    }

    pub fn total_payload_bytes(&self) -> usize {
        self.raw.len()
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
