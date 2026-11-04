use std::time::Instant;

/// A tiny, harmless 2-input / 3-class linear classifier.
/// These are quantized toy parameters, not weights from a real model.
const SCALE: f32 = 100.0;
const WEIGHTS_Q: [i16; 6] = [120, -40, -30, 110, 20, 15];
const BIASES_Q: [i16; 3] = [5, -3, 1];

#[derive(Debug)]
struct ToyModel {
    weights: Vec<f32>,
    biases: Vec<f32>,
}

impl ToyModel {
    /// Reconstruct floating-point tensors from source-embedded integer values.
    fn reconstruct() -> Self {
        let started = Instant::now();

        let weights = WEIGHTS_Q
            .iter()
            .map(|&value| value as f32 / SCALE)
            .collect();

        let biases = BIASES_Q
            .iter()
            .map(|&value| value as f32 / SCALE)
            .collect();

        eprintln!(
            "[monitor] event=model_reconstructed elapsed_us={}",
            started.elapsed().as_micros()
        );

        Self { weights, biases }
    }

    /// Benign inference: classify a simple two-number input.
    fn predict(&self, input: [f32; 2]) -> (usize, [f32; 3]) {
        let mut scores = [0.0_f32; 3];

        for class_idx in 0..3 {
            let row = class_idx * 2;
            scores[class_idx] = self.weights[row] * input[0]
                + self.weights[row + 1] * input[1]
                + self.biases[class_idx];
        }

        let mut best = 0;
        for idx in 1..scores.len() {
            if scores[idx] > scores[best] {
                best = idx;
            }
        }

        (best, scores)
    }

    /// Best-effort overwrite of owned buffers before they are dropped.
    ///
    /// This is NOT a forensic erasure guarantee: copies may exist in registers,
    /// compiler temporaries, allocator pages, swap, or crash dumps.
    fn best_effort_cleanup(&mut self) {
        self.weights.fill(0.0);
        self.biases.fill(0.0);
        eprintln!("[monitor] event=cleanup_requested buffers_overwritten=true");
    }
}

fn main() {
    println!("Ephemeral SLM research demo (safe toy model)");
    println!("No network access, exploitation, persistence, or self-deletion.");

    let mut model = ToyModel::reconstruct();

    // Two benign sample features. The output classes are arbitrary labels.
    let sample = [0.7_f32, 0.2_f32];
    let (class_id, scores) = model.predict(sample);

    println!("Input features: {sample:?}");
    println!("Class scores:   {scores:?}");
    println!("Predicted class: {class_id}");

    model.best_effort_cleanup();
    println!("Finished. Process exits normally; source files remain intact.");
}
