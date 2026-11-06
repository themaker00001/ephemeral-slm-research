//! Ephemeral SLM Research - Safe toy demo of source-embedded Small Language Models,
//! runtime reconstruction, layer-by-layer ephemeral execution, and volatile memory scrubbing.

pub mod ephemeral;
pub mod forensics;
pub mod model;
pub mod telemetry;
pub mod weights;

pub use ephemeral::{volatile_zero, EphemeralMetrics, SecureBuffer};
pub use model::{EphemeralSLM, MathOps, MonolithicSLM};
pub use telemetry::{get_process_rss_bytes, LifecycleEvent, TelemetryLogger};
pub use weights::{ModelConfig, PayloadView, EMBEDDED_MODEL_PAYLOAD};
