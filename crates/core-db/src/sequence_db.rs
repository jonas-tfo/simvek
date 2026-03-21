
use std::path::Path;
use anyhow::Ok;
use anyhow::{Result, Context};
use crate::types::Storage;
use crate::types::FastaRecord;

impl Storage {
    pub fn open(path: &Path) -> Result<Self> {
        let db: sled::Db = sled::open(path)?;
        let records = db.open_tree("records")?; // creates or opens table in db
        Ok(Self {db: db, records: records})
    }

    pub fn insert(&self, record: &FastaRecord) -> Result<u64> {
        let id = self.db.generate_id()?;
        let id_ser: [u8;_] = id.to_be_bytes(); // big endian because keys stored sorted
        let record_ser: Vec<u8> = bincode::serialize(&record)?;
        self.records.insert(id_ser, record_ser)?;
        Ok(id)
    }

    pub fn get(&self, internal_id: u64) -> Result<Option<FastaRecord>> {
        let id_be = internal_id.to_be_bytes();
        let record_ser = self.records.get(id_be)?;
        match record_ser {
            None => Ok(None),
            Some(bytes) => {
                let record = bincode::deserialize(&bytes).expect("Failed to deserialize retrieved record");
                Ok(Some(record))}
        }
    }

    pub fn delete(&self, internal_id: u64) -> Result<()> {
        let id_be = internal_id.to_be_bytes();
        let deleted_val = self.records.remove(id_be)?;
        match deleted_val {
            None => {
                eprintln!("warning: no record found for id {}", internal_id);
            }
            Some(_) => {}
        }
        Ok(())
    }

    pub fn iter(&self) -> impl Iterator<Item = Result<(u64, FastaRecord)>> + '_ {
        self.records.iter().map(|item| {
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
    pub fn flush(&self) -> Result<usize> {
        let written = self.db.flush()?;
        Ok(written)
    }

}

