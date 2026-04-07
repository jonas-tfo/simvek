use std::path::{Path, PathBuf};
use std::process::Command;
use anyhow::{Context, Result, bail};
use core_db::types::{ModelSignature, SequenceEmbedder};
use rand::seq::SliceRandom;

pub struct PythonEmbedder {
    script_path: PathBuf,
    model_name: String,
    dimension: usize,
}

impl PythonEmbedder {
    pub fn new(script_path: &Path, model_name: &str, dimension: usize) -> Self {
        Self {
            script_path: script_path.to_path_buf(),
            model_name: model_name.to_string(),
            dimension,
        }
    }
}

impl SequenceEmbedder for PythonEmbedder {

    fn embed(&self, sequence: &[u8]) -> Result<Vec<f32>> {
        let sequence_str = std::str::from_utf8(sequence)
            .context("sequence is not valid utf-8")?;

        let output = Command::new("python3")
            .arg(&self.script_path)
            .arg("--sequence").arg(sequence_str)
            .arg("--model").arg(&self.model_name)
            .output()
            .context("failed to run python embedder")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            bail!("python embedder failed: {}", stderr);
        }

        let stdout = String::from_utf8(output.stdout)
            .context("python output was not valid utf-8")?;

        let embedding: Vec<f32> = serde_json::from_str(&stdout)
            .context("failed to parse embedding from python output")?;

        if embedding.len() != self.dimension {
            bail!(
                "expected embedding dimension {}, got {}",
                self.dimension,
                embedding.len()
            );
        }

        Ok(embedding)
    }
    fn embed_dev(&self, _sequence: &[u8])-> Result<Vec<f32>> {
        let mut rng = rand::rng();
        let mut nums: Vec<f32> = (0..1024).map(|i| i as f32).collect();
        nums.shuffle(&mut rng);

        Ok(nums)
    }

    fn embed_batch(&self, fasta: &PathBuf) -> Result<Vec<Vec<f32>>> {
        let output = Command::new("python3")
            .arg(&self.script_path)
            .arg("--fasta").arg(fasta.to_str().context("invalid path")?)
            .arg("--model").arg(&self.model_name)
            .output()
            .context("failed to run python embedder")?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            bail!("python embedder failed: {}", stderr);
        }
        let stdout = String::from_utf8(output.stdout)
            .context("python output was not valid utf-8")?;
        let embeddings: Vec<Vec<f32>> = serde_json::from_str(&stdout)
            .context("failed to parse embedding from python output")?;
        Ok(embeddings)
    }

    fn get_dimension(&self) -> usize {
        self.dimension
    }

    fn get_signature(&self) -> ModelSignature {
        ModelSignature {
            name: self.model_name.clone(),
            dimension: self.dimension,
        }
    }
}
