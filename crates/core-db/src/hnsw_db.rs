
use anyhow::{Context, Ok, Result};
use std::path::PathBuf;
use hnsw_rs::hnsw::Neighbour;
use hnsw_rs::prelude::{Hnsw, DistL2};
use crate::types::{FastaRecord, HnswDB, HnswDBConfig, HnswSearchQuery, SeqType, SledDB, SequenceEmbedder};
use crate::fileutils::parse_fasta;


impl HnswDBConfig {
    pub fn default(sequence_sled_data_path: PathBuf, vector_sled_data_path: PathBuf, record_type: SeqType) -> Self {
        Self {
            sequence_sled_data_path,
            vector_sled_data_path,
            ef_construction: 200,
            max_nb_connection: 16,
            expected_size: 100_000,
            ef_search: 50,
            max_layers: 16,
            record_type: record_type
        }
    }
}

impl<'a> HnswSearchQuery<'a> {
    pub fn default(vec: &'a [u8]) -> Self {
        Self {
            data: vec,
            knn: 5,
            search_width: 10,
        }
    }
}

impl HnswDB {

    pub fn open(config: HnswDBConfig, embedder: Box<dyn SequenceEmbedder>) -> Result<Self> {
        let sequence_db = SledDB::open(&config.sequence_sled_data_path).context("failed to open sequence db storage")?;
        let vector_db = SledDB::open(&config.vector_sled_data_path).context("failed to open sequence db storage")?;
        // make new hnsw db
        let hnsw_storage = Hnsw::new(
            config.max_nb_connection,
            config.expected_size,
            config.max_layers,
            config.ef_construction,
            DistL2::default(),
        );

        let mut db = Self {
            sequence_db,
            vector_db,
            hnsw_storage,
            embedder,
            config,
        };
        // rebuild if records exist
        db.rebuild_index().context("failed to rebuild HNSW index from storage")?;
        Ok(db)
    }

    pub fn open_fresh(config: HnswDBConfig, embedder: Box<dyn SequenceEmbedder>) -> Result<Self> {
        let sequence_db = SledDB::open(&config.sequence_sled_data_path).context("failed to open sequence storage")?;
        let vector_db = SledDB::open(&config.vector_sled_data_path).context("failed to open vector storage")?;
        sequence_db.clear()?;
        vector_db.clear()?;
        // make new hnsw db
        let hnsw_storage = Hnsw::new(
            config.max_nb_connection,
            config.expected_size,
            config.max_layers,
            config.ef_construction,
            DistL2::default(),
        );

        let db = Self {
            sequence_db,
            vector_db,
            hnsw_storage,
            embedder,
            config,
        };
        Ok(db)
    }

    /// fill database from a fasta file
    pub fn rebuild_index_from_fasta(&mut self, fasta: &PathBuf) -> Result<()> {
        let (ids, seqs, seq_type) = parse_fasta(fasta)?;
        for (id, seq) in ids.iter().zip(seqs.iter()) {
            let record = FastaRecord {
                header: id.to_string(),
                sequence: seq.clone().into_bytes(),
                seq_type: seq_type
            };
            self.insert(record)?;
        }
        Ok(())
    }

    pub fn rebuild_index_from_fasta_batch(&mut self, fasta: &PathBuf) -> Result<()> {
        self.insert_batch(fasta)?;
        Ok(())
    }


    /// rebuild HNSW index from stored embeddings in vector_db
    fn rebuild_index(&mut self) -> Result<()> {
        let mut count = 0;
        for entry in self.vector_db.iter_embeddings() {
            let (internal_id, embedding) = entry.context("failed to read embedding during index rebuild")?;
            self.hnsw_storage.insert((&embedding, internal_id as usize));
            count += 1;
        }
        if count > 0 {
            println!("rebuilt HNSW index from {} records", count);
        }
        Ok(())
    }

    /// store in sled, embed sequence, store embedding in vector db
    pub fn insert(&mut self, record: FastaRecord) -> Result<u64> {
        let embedding: Vec<f32> = self.embedder.embed(&record.sequence).context("Failed to embed the sequence")?;
        let internal_id = self.sequence_db.insert(&record).context("Failed to insert record into sequence db")?;
        self.vector_db.insert_embedding(internal_id, &embedding).context("Failed to store embedding in vector db")?;
        self.hnsw_storage.insert((&embedding, internal_id as usize));
        Ok(internal_id)
    }

    pub fn insert_batch(&mut self, fasta: &PathBuf) -> Result<Vec<u64>> {
        let (ids, seqs, seq_type) = parse_fasta(&fasta)?;
        let embeddings: Vec<Vec<f32>> = self.embedder.embed_batch(&fasta)?;
        let mut internal_ids: Vec<u64> = Vec::with_capacity(ids.len());
        for ((id, seq), embedding) in ids.into_iter().zip(seqs).zip(embeddings) {
            let record = FastaRecord {
                header: id,
                sequence: seq.into_bytes(),
                seq_type: seq_type
            };
            let internal_id = self.sequence_db.insert(&record).context("Failed to insert record")?;
            self.vector_db.insert_embedding(internal_id, &embedding).context("Failed to store embedding")?;
            self.hnsw_storage.insert((embedding.as_slice(), internal_id as usize));
            internal_ids.push(internal_id);
        }
        Ok(internal_ids)
    }

    /// get kNN and give nearest records and their l2 distance to query
    pub fn search(&self, query: HnswSearchQuery) -> Result<Vec<Neighbour>> {
        let embedding: Vec<f32> = self.embedder.embed(&query.data)
            .context("Failed to embed the sequence")?;
        let neighbours: Vec<Neighbour> = self.hnsw_storage.search(&embedding, query.knn, query.search_width);
        Ok(neighbours)
    }

    /// remove from sled, cant remove from hnsw though, always dead vector?
    pub fn delete(&mut self, internal_id: u64, seq_type: SeqType) -> Result<()> {
        self.sequence_db.delete(internal_id, seq_type)?;
        Ok(())
    }

    pub fn clear(&mut self) -> Result<()> {
        self.sequence_db.db.clear()?;
        Ok(())
    }

    /// save hnsw index to disk
    pub fn save_index(&self) -> Result<()> {
        todo!()
    }


}
