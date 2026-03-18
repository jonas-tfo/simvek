
use anyhow::{Result, Context};
use ort::session::Session;
use std::path::Path;

pub struct OnnxSession {
    pub session: Session,
}

impl OnnxSession {
    pub fn load(model_path: &Path) -> Result<Self> {
        let session = Session::builder()
            .context("failed to create session builder")?
            .commit_from_file(model_path)
            .context("failed to load onnx model")?;

        Ok(Self { session })
    }
}
