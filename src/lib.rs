//! Ephemeral SLM Research - a testbed for keeping a small language model encrypted in the
//! executable, decrypting and dequantizing it one layer at a time, wiping each layer after
//! use, and checking process memory for leftovers.

pub mod ephemeral;
pub mod forensics;
pub mod model;
pub mod seal;
pub mod telemetry;
pub mod weights;

pub use ephemeral::{volatile_zero, EphemeralMetrics, SecureBuffer};
pub use model::{EphemeralSLM, MathOps, MonolithicSLM};
pub use telemetry::{get_process_rss_bytes, LifecycleEvent, TelemetryLogger};
pub use seal::{SealError, SealedPayload, SecretKey};
pub use weights::{ModelConfig, PayloadView, EMBEDDED_SEALED_PAYLOAD};
