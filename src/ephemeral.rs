//! Ephemeral memory management, volatile scrubbing, and defensive audits.
//!
//! Unlike standard `Vec::fill(0.0)` which LLVM dead-code elimination (DCE) can
//! freely eliminate when a buffer is dropped immediately afterward, `SecureBuffer`
//! uses volatile writes and full memory barriers (`compiler_fence`) to guarantee
//! that parameters are physically overwritten in RAM.

use std::ops::{Deref, DerefMut};
use std::sync::atomic::{compiler_fence, Ordering};

/// Overwrite every byte of `data` with zero using volatile stores that the compiler may
/// not elide, followed by a compiler fence.
///
/// This is the single zeroization primitive used by [`SecureBuffer`] and by the
/// monolithic baseline's secure cleanup. It only covers the bytes of `data` itself:
/// copies made earlier (registers, stack temporaries, allocator metadata, swap) are
/// out of its reach.
pub fn volatile_zero<T: Copy>(data: &mut [T]) {
    if data.is_empty() {
        return;
    }
    compiler_fence(Ordering::SeqCst);
    let ptr = data.as_mut_ptr() as *mut u8;
    let total_bytes = std::mem::size_of_val(data);
    // SAFETY: `ptr` points to `total_bytes` initialized, writable bytes owned by `data`.
    // Zero is written as raw bytes; `T: Copy` means no destructor can observe the value.
    unsafe {
        for i in 0..total_bytes {
            std::ptr::write_volatile(ptr.add(i), 0u8);
        }
    }
    compiler_fence(Ordering::SeqCst);
}

/// A memory buffer that guarantees secure zeroization upon drop or manual scrubbing.
pub struct SecureBuffer<T: Copy> {
    data: Vec<T>,
    is_scrubbed: bool,
}

impl<T: Default + Copy> SecureBuffer<T> {
    /// Allocate a new zero-initialized buffer of length `len`.
    pub fn zeroed(len: usize) -> Self {
        let mut data = Vec::with_capacity(len);
        data.resize(len, T::default());
        Self {
            data,
            is_scrubbed: false,
        }
    }

    /// Allocate a buffer from an existing `Vec<T>`.
    pub fn from_vec(data: Vec<T>) -> Self {
        Self {
            data,
            is_scrubbed: false,
        }
    }

    /// The number of elements in the buffer.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Is the buffer empty?
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Total size in bytes.
    #[inline]
    pub fn size_bytes(&self) -> usize {
        self.data.len() * std::mem::size_of::<T>()
    }

    /// Whether the buffer has already undergone secure zeroization.
    #[inline]
    pub fn is_scrubbed(&self) -> bool {
        self.is_scrubbed
    }

    /// Execute a secure volatile zeroization of the buffer's contents.
    ///
    /// Uses `core::ptr::write_volatile` and `compiler_fence(Ordering::SeqCst)`
    /// so the compiler cannot optimize out or reorder the clearing operation.
    pub fn scrub(&mut self) {
        if self.is_scrubbed || self.data.is_empty() {
            return;
        }
        volatile_zero(&mut self.data);
        self.is_scrubbed = true;
    }

    /// Forensic inspection: inspect every byte of the underlying memory to verify
    /// that no residual weights, non-zero floats, or recognizable signatures remain.
    pub fn audit_is_zero(&self) -> bool {
        let ptr = self.data.as_ptr() as *const u8;
        let total_bytes = self.size_bytes();
        let mut all_zero = true;

        unsafe {
            for i in 0..total_bytes {
                let byte = std::ptr::read_volatile(ptr.add(i));
                if byte != 0 {
                    all_zero = false;
                    break;
                }
            }
        }
        all_zero
    }
}

impl<T: Copy> Deref for SecureBuffer<T> {
    type Target = [T];
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<T: Copy> DerefMut for SecureBuffer<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.is_scrubbed = false;
        &mut self.data
    }
}

impl<T: Copy> Drop for SecureBuffer<T> {
    fn drop(&mut self) {
        // Zeroize on drop if not explicitly scrubbed earlier.
        if !self.is_scrubbed {
            volatile_zero(&mut self.data);
            self.is_scrubbed = true;
        }
    }
}

/// Real-time metrics tracker for ephemeral memory allocations and lifetime scrubbing.
#[derive(Debug, Default, Clone)]
pub struct EphemeralMetrics {
    pub total_allocations: usize,
    pub total_bytes_allocated: usize,
    pub total_bytes_scrubbed: usize,
    pub current_active_bytes: usize,
    pub peak_active_bytes: usize,
    pub scrub_operations: usize,
    pub audits_passed: usize,
    pub audits_failed: usize,
    /// Encrypted sections that were decrypted (each into its own transient buffer).
    pub sections_opened: usize,
    /// Bytes of decrypted INT8 plaintext that were wiped after dequantization.
    pub plaintext_bytes_wiped: usize,
    /// Largest decrypted INT8 section held at one time.
    pub peak_plaintext_bytes: usize,
}

impl EphemeralMetrics {
    /// Record that a decrypted INT8 section of `bytes` was wiped, and whether the
    /// read-back audit found it all zero. Counted in the same audit totals as the f32 buffers.
    pub fn record_plaintext_wipe(&mut self, bytes: usize, clean: bool) {
        self.sections_opened += 1;
        self.plaintext_bytes_wiped += bytes;
        if bytes > self.peak_plaintext_bytes {
            self.peak_plaintext_bytes = bytes;
        }
        self.record_audit(clean);
    }

    pub fn record_allocation(&mut self, bytes: usize) {
        self.total_allocations += 1;
        self.total_bytes_allocated += bytes;
        self.current_active_bytes += bytes;
        if self.current_active_bytes > self.peak_active_bytes {
            self.peak_active_bytes = self.current_active_bytes;
        }
    }

    pub fn record_scrub(&mut self, bytes: usize) {
        self.scrub_operations += 1;
        self.total_bytes_scrubbed += bytes;
        self.current_active_bytes = self.current_active_bytes.saturating_sub(bytes);
    }

    pub fn record_audit(&mut self, passed: bool) {
        if passed {
            self.audits_passed += 1;
        } else {
            self.audits_failed += 1;
        }
    }
}
