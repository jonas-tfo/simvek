
use std::path::Path;
use anyhow::Result;
use core_engine::types::{ModelSignature, SequenceEmbedder};
use crate::onnx::OnnxSession;

pub struct ProtT5 {
    session: OnnxSession,
    dimension: usize,
}

impl ProtT5 {
    pub fn load(model_path: &Path) -> Result<Self> {
        let session = OnnxSession::load(model_path)?;
        Ok(Self {
            session,
            dimension: 1024,  // ouput dim for prot t5
        })
    }
}

impl SequenceEmbedder for ProtT5 {
    fn embed(&self, sequence: &[u8]) -> Result<Vec<f32>> {
        // random ve for testing
        Ok(vec![0.0f32; self.dimension])
    }

    fn embed_batch(&self, sequences: &[&[u8]]) -> Result<Vec<Vec<f32>>> {
        todo!()
    }

    fn get_dimension(&self) -> usize {
        self.dimension
    }

    fn get_signature(&self) -> ModelSignature {
        ModelSignature {
            name: "prot_t5".to_string(),
            dimension: self.dimension,
        }
    }
}
