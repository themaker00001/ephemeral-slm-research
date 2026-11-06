//! Defensive telemetry, OS-level memory monitoring, and event logging.
//!
//! Queries the host operating system kernel (Mach on macOS, /proc on Linux)
//! to obtain exact, real-time Resident Set Size (RSS) and virtual memory metrics.

use std::time::Instant;

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Default, Debug, Clone, Copy)]
pub struct MachTaskBasicInfo {
    pub virtual_size: u64,
    pub resident_size: u64,
    pub resident_size_max: u64,
    pub user_time_sec: i32,
    pub user_time_usec: i32,
    pub system_time_sec: i32,
    pub system_time_usec: i32,
    pub policy: i32,
    pub suspend_count: i32,
}

#[cfg(target_os = "macos")]
extern "C" {
    fn mach_task_self() -> u32;
    fn task_info(
        target_task: u32,
        flavor: i32,
        task_info_out: *mut MachTaskBasicInfo,
        task_info_out_cnt: *mut u32,
    ) -> i32;
}

/// Retrieve the current process Resident Set Size (RSS) in bytes directly from the OS kernel.
pub fn get_process_rss_bytes() -> u64 {
    #[cfg(target_os = "macos")]
    {
        let mut info = MachTaskBasicInfo::default();
        let mut count = (std::mem::size_of::<MachTaskBasicInfo>() / std::mem::size_of::<u32>()) as u32;
        let kr = unsafe {
            task_info(mach_task_self(), 20, &mut info as *mut _, &mut count as *mut _)
        };
        if kr == 0 {
            info.resident_size
        } else {
            0
        }
    }

    #[cfg(target_os = "linux")]
    {
        if let Ok(statm) = std::fs::read_to_string("/proc/self/statm") {
            let parts: Vec<&str> = statm.split_whitespace().collect();
            if parts.len() >= 2 {
                if let Ok(pages) = parts[1].parse::<u64>() {
                    return pages * 4096;
                }
            }
        }
        0
    }

    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        0
    }
}

/// Structured events emitted during the lifecycle of an ephemeral SLM.
#[derive(Debug, Clone)]
pub enum LifecycleEvent<'a> {
    ModelInitialized {
        mode: &'a str,
        total_parameters: usize,
        payload_bytes: usize,
        initial_rss_kb: u64,
    },
    LayerReconstructed {
        layer_idx: usize,
        tensor_bytes: usize,
        elapsed_us: u128,
        rss_kb: u64,
    },
    LayerScrubbed {
        layer_idx: usize,
        scrubbed_bytes: usize,
        audit_zeroed: bool,
        rss_kb: u64,
    },
    TokenStep {
        step: usize,
        token: u8,
        token_char: char,
        elapsed_us: u128,
        active_memory_kb: usize,
    },
    CleanupCompleted {
        mode: &'a str,
        total_scrubbed_bytes: usize,
        audits_passed: usize,
        final_rss_kb: u64,
    },
}

pub struct TelemetryLogger {
    pub verbose: bool,
    start_time: Instant,
}

impl TelemetryLogger {
    pub fn new(verbose: bool) -> Self {
        Self {
            verbose,
            start_time: Instant::now(),
        }
    }

    pub fn log(&self, event: LifecycleEvent) {
        let elapsed_ms = self.start_time.elapsed().as_millis();
        match event {
            LifecycleEvent::ModelInitialized {
                mode,
                total_parameters,
                payload_bytes,
                initial_rss_kb,
            } => {
                eprintln!(
                    "[monitor] t=+{}ms event=model_initialized mode={} params={} payload_bytes={} init_rss_kb={}",
                    elapsed_ms, mode, total_parameters, payload_bytes, initial_rss_kb
                );
            }
            LifecycleEvent::LayerReconstructed {
                layer_idx,
                tensor_bytes,
                elapsed_us,
                rss_kb,
            } => {
                if self.verbose {
                    eprintln!(
                        "[monitor] t=+{}ms event=layer_reconstructed layer={} bytes={} reconstruct_us={} rss_kb={}",
                        elapsed_ms, layer_idx, tensor_bytes, elapsed_us, rss_kb
                    );
                }
            }
            LifecycleEvent::LayerScrubbed {
                layer_idx,
                scrubbed_bytes,
                audit_zeroed,
                rss_kb,
            } => {
                if self.verbose {
                    eprintln!(
                        "[monitor] t=+{}ms event=layer_scrubbed layer={} bytes={} audit={} rss_kb={}",
                        elapsed_ms,
                        layer_idx,
                        scrubbed_bytes,
                        if audit_zeroed { "VERIFIED_ZERO" } else { "FAILED_LEAK" },
                        rss_kb
                    );
                }
            }
            LifecycleEvent::TokenStep {
                step,
                token,
                token_char,
                elapsed_us,
                active_memory_kb,
            } => {
                if self.verbose {
                    let printable = if token_char.is_ascii_graphic() || token_char == ' ' {
                        token_char
                    } else {
                        '·'
                    };
                    eprintln!(
                        "[monitor] t=+{}ms event=token_step step={:02} token={} char='{}' active_kb={} us={}",
                        elapsed_ms, step, token, printable, active_memory_kb, elapsed_us
                    );
                }
            }
            LifecycleEvent::CleanupCompleted {
                mode,
                total_scrubbed_bytes,
                audits_passed,
                final_rss_kb,
            } => {
                eprintln!(
                    "[monitor] t=+{}ms event=cleanup_completed mode={} total_scrubbed_bytes={} audits_passed={} final_rss_kb={}",
                    elapsed_ms, mode, total_scrubbed_bytes, audits_passed, final_rss_kb
                );
            }
        }
    }
}
