
use std::path::Path;
use anyhow::Ok;
use anyhow::{Result, Context};
use crate::SeqType;
use crate::types::SledDB;
use crate::types::FastaRecord;

impl SledDB {
    pub fn open(path: &Path) -> Result<Self> {
        let db: sled::Db = sled::open(path)?;
        Ok(Self {
            dna_records: db.open_tree("dna")?,
            rna_records: db.open_tree("rna")?,
            protein_records: db.open_tree("protein")?,
            embeddings: db.open_tree("embeddings")?,
            db,
        })
    }

    pub fn get_tree_type(&self, seq_type: SeqType) -> &sled::Tree {
        match seq_type {
            SeqType::Dna => &self.dna_records,
            SeqType::Rna => &self.rna_records,
            SeqType::Protein => &self.protein_records
        }
    }

    pub fn insert(&self, record: &FastaRecord) -> Result<u64> {
        let id = self.db.generate_id()?;
        let id_ser: [u8;_] = id.to_be_bytes(); // big endian because keys stored sorted
        let record_ser: Vec<u8> = bincode::serialize(&record)?;
        self.get_tree_type(record.seq_type).insert(id_ser, record_ser)?;
        Ok(id)
    }

    pub fn insert_with_id(&self, id: u64, record: &FastaRecord) -> Result<()> {
        let id_ser: [u8; 8] = id.to_be_bytes();
        let record_ser: Vec<u8> = bincode::serialize(&record)?;
        self.get_tree_type(record.seq_type).insert(id_ser, record_ser)?;
        Ok(())
    }

    pub fn get(&self, internal_id: u64, seq_type: SeqType) -> Result<Option<FastaRecord>> {
        let id_be = internal_id.to_be_bytes();
        let record_ser = self.get_tree_type(seq_type).get(id_be)?;
        match record_ser {
            None => Ok(None),
            Some(bytes) => {
                let record = bincode::deserialize(&bytes).expect("Failed to deserialize retrieved record");
                Ok(Some(record))}
        }
    }

    pub fn delete(&self, internal_id: u64, seq_type: SeqType) -> Result<()> {
        let id_be = internal_id.to_be_bytes();
        let deleted_val = self.get_tree_type(seq_type).remove(id_be)?;
        match deleted_val {
            None => {
                eprintln!("warning: no record found for id {}", internal_id);
            }
            Some(_) => {}
        }
        Ok(())
    }

    pub fn iter(&self, seq_type: SeqType) -> impl Iterator<Item = Result<(u64, FastaRecord)>> + '_ {
        self.get_tree_type(seq_type).iter().map(|item| {
            let (key, value) = item.context("failed to read from sled")?;

            let internal_id = u64::from_be_bytes(
                key.as_ref()
                    .try_into()
                    .context("failed to convert key to u64")?
            );

            let record: FastaRecord = bincode::deserialize(&value)
                .context("failed to deserialize record")?;

            Ok((internal_id, record))
        })
    }
    pub fn insert_embedding(&self, id: u64, embedding: &[f32]) -> Result<()> {
        let id_ser: [u8; 8] = id.to_be_bytes();
        let emb_ser = bincode::serialize(embedding)?;
        self.embeddings.insert(id_ser, emb_ser)?;
        Ok(())
    }

    pub fn get_embedding(&self, id: u64) -> Result<Option<Vec<f32>>> {
        let id_ser = id.to_be_bytes();
        match self.embeddings.get(id_ser)? {
            None => Ok(None),
            Some(bytes) => Ok(Some(bincode::deserialize(&bytes)?)),
        }
    }

    pub fn iter_embeddings(&self) -> impl Iterator<Item = Result<(u64, Vec<f32>)>> + '_ {
        self.embeddings.iter().map(|item| {
            let (key, value) = item.context("failed to read embedding from sled")?;
            let id = u64::from_be_bytes(key.as_ref().try_into().context("bad embedding key")?);
            let embedding: Vec<f32> = bincode::deserialize(&value).context("failed to deserialize embedding")?;
            Ok((id, embedding))
        })
    }

    pub fn clear(&self) -> Result<()> {
        self.dna_records.clear()?;
        self.rna_records.clear()?;
        self.protein_records.clear()?;
        self.embeddings.clear()?;
        Ok(())
    }

    pub fn flush(&self) -> Result<usize> {
        let written = self.db.flush()?;
        Ok(written)
    }

}

