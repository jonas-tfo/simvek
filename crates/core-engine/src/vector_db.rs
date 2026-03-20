
use anyhow::{Context, Ok, Result};
use std::path::PathBuf;
use hnsw_rs::hnsw::Neighbour;
use hnsw_rs::prelude::{Hnsw, DistL2};
use crate::types::{Storage, FastaRecord, HnswSearchQuery, SequenceEmbedder, VectorDB, VectorDBConfig};


impl VectorDBConfig {
    pub fn default(path: PathBuf) -> Self {
        Self {
            path,
            ef_construction: 200,
            max_nb_connection: 16,
            expected_size: 100_000,
            ef_search: 50,
            max_layers: 16,
        }
    }
}

impl VectorDB {

    pub fn open(config: VectorDBConfig, embedder: Box<dyn SequenceEmbedder>) -> Result<Self> {
        let sled_storage = Storage::open(&config.path)
            .context("failed to open sled storage")?;
        // make new hnsw db
        let hnsw_storage = Hnsw::new(
            config.max_nb_connection,
            config.expected_size,
            config.max_layers,
            config.ef_construction,
            DistL2::default(),
        );

        let mut db = Self {
            sled_storage,
            hnsw_storage,
            embedder,
            config,
        };
        // rebuild if records exist
        db.rebuild_index()
            .context("failed to rebuild HNSW index from storage")?;
        Ok(db)
    }

    fn rebuild_index(&mut self) -> Result<()> {
        let mut count = 0;
        for entry in self.sled_storage.iter() {
            let (internal_id, record) = entry
                .context("failed to read record during index rebuild")?;

            let embedding = self.embedder.embed(&record.sequence)
                .context("failed to embed sequence during rebuild")?;

            self.hnsw_storage.insert((&embedding, internal_id as usize));
            count += 1;
        }
        if count > 0 {
            println!("rebuilt HNSW index from {} records", count);
        }
        Ok(())
    }

    /// store in sled, embed sequence, store in vector db
    pub fn insert(&mut self, record: FastaRecord) -> Result<u64> {
        let embedding: Vec<f32> = self.embedder.embed(&record.sequence).context("Failed to embed the sequence")?;
        let internal_id = self.sled_storage.insert(&record).context("Failed to insert record into db")?;
        self.hnsw_storage.insert((&embedding, internal_id as usize));
        Ok(internal_id)
    }

    pub fn insert_batch(&mut self, records: Vec<FastaRecord>) -> Result<Vec<u64>> {
        todo!()
    }

    /// get kNN and give nearest records and their l2 distance to query
    pub fn search(&self, query: HnswSearchQuery) -> Result<Vec<Neighbour>> {
        let embedding: Vec<f32> = self.embedder.embed(&query.data)
            .context("Failed to embed the sequence")?;
        let neighbours: Vec<Neighbour> = self.hnsw_storage.search(&embedding, query.knn, query.search_width);
        Ok(neighbours)
    }

    /// remove from sled, cant remove from hnsw though, always dead vector?
    pub fn delete(&mut self, internal_id: u64) -> Result<()> {
        self.sled_storage.delete(internal_id)?;
        Ok(())
    }

    /// save hnsw index to disk
    pub fn save_index(&self) -> Result<()> {
        todo!()
    }

}
