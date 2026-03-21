
use std::path::{PathBuf};
use serde::{Serialize, Deserialize};
use hnsw_rs::prelude::{Hnsw, DistL2};
use anyhow::Result;


#[derive(Serialize, Deserialize, PartialEq, Debug)]
pub struct ModelSignature {
    pub name: String,
    pub dimension: usize,
}

pub trait SequenceEmbedder {
    fn embed(&self, sequence: &[u8]) -> Result<Vec<f32>>;
    fn embed_batch(&self, sequences: &[&[u8]]) -> Result<Vec<Vec<f32>>>;
    fn get_dimension(&self) -> usize;
    fn get_signature(&self) -> ModelSignature;
}


#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
pub enum SeqType {
    Dna,
    Rna,
    Protein,
}

/// record for the sled storage
#[derive(Serialize, Deserialize, Debug)]
pub struct FastaRecord {
    pub header: String,
    pub sequence: Vec<u8>,  // raw seq as vec of bytes
    pub seq_type: SeqType,
}

pub struct Storage {
    pub db: sled::Db,
    pub records: sled::Tree,  // tree for FastaRecord entries
}

pub struct VectorDBConfig {
    pub path: PathBuf,
    pub ef_construction: usize,  // build accuracy
    pub max_nb_connection: usize,// graph connectivity
    pub expected_size: usize,    // rough upper bound number of sequences
    pub ef_search: usize,        // search accuracy
    pub max_layers: usize,       // hnsw layer count
}

pub struct VectorDB {
    pub sled_storage: Storage,
    pub hnsw_storage: Hnsw<'static, f32, DistL2>,
    pub embedder: Box<dyn SequenceEmbedder>,
    pub config: VectorDBConfig
}

// data cant outlive search query
pub struct HnswSearchQuery<'a> {
    pub data: &'a [u8],
    pub knn: usize,
    pub search_width: usize
}

