use std::path::{Path, PathBuf};
use std::process::Command;
use anyhow::{Context, Result, bail};
use core_db::types::{ModelSignature, SequenceEmbedder};
use rand::seq::SliceRandom;

pub struct OnnxEmbedder {
    onnx_path: PathBuf,
    model_name: String,
    dimension: usize,
}

impl OnnxEmbedder {
    fn new(script_path: &Path, model_name: &str, dimension: usize) -> Self {
        Self {
            onnx_path: script_path.to_path_buf(),
            model_name: model_name.to_string(),
            dimension,
        }
    }
}

impl SequenceEmbedder for OnnxEmbedder {

    fn embed(&self, sequence: &[u8]) -> Result<Vec<f32>> {
        todo!()
    }

    fn embed_batch(&self, sequences: &PathBuf) -> Result<Vec<Vec<f32>>> {
        todo!()
    }

    fn embed_dev(&self, sequence: &[u8]) -> Result<Vec<f32>> {
        todo!()
    }

    fn get_dimension(&self) -> usize {
        todo!()
    }

    fn get_signature(&self) -> ModelSignature {
        todo!()
    }
}

