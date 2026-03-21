
#[cfg(test)]
mod tests {
    use core_db::types::{FastaRecord, SeqType, Storage};
    use tempfile::tempdir;

    #[test]
    fn insert_and_retrieve() {
        let temp = tempdir().unwrap();
        let storage = Storage::open(temp.path()).unwrap();
        let record = FastaRecord {
            header: String::from(">1jon"),
            sequence: String::from("AAAAATTTTTTCCCCCCCCCGGGGGGGG").into_bytes().to_vec(),
            seq_type: SeqType::Dna
        };
        let internal_id = storage.insert(&record).unwrap();
        let retrieved = storage.get(internal_id).unwrap().unwrap();

        assert_eq!(record.header, retrieved.header);
        assert_eq!(record.sequence, retrieved.sequence);
    }

    #[test]
    fn ids_are_monotonically_increasing() {
        let temp = tempdir().unwrap();
        let storage = Storage::open(temp.path()).unwrap();
        let record1 = FastaRecord {
            header: String::from(">1jon"),
            sequence: String::from("AAAAATTTTTTCCCCCCCCCGGGGGGGG").into_bytes().to_vec(),
            seq_type: SeqType::Dna
        };
        let record2 = FastaRecord {
            header: String::from(">1jona"),
            sequence: String::from("AAAAATTTTTTCCCCCCCCCGGGGGGGG").into_bytes().to_vec(),
            seq_type: SeqType::Dna
        };
        let record3 = FastaRecord {
            header: String::from(">1jonas"),
            sequence: String::from("AAAAATTTTTTCCCCCCCCCGGGGGGGG").into_bytes().to_vec(),
            seq_type: SeqType::Dna
        };
        let id1 = storage.insert(&record1).unwrap();
        let id2 = storage.insert(&record2).unwrap();
        let id3 = storage.insert(&record3).unwrap();
        assert!(id1 < id2 && id2 < id3);
    }

    #[test]
    fn survives_reopen() {
        let temp = tempdir().unwrap();
        let record = FastaRecord {
            header: String::from(">1jon"),
            sequence: b"AAAAATTTTTTCCCCCCCCCGGGGGGGG".to_vec(),
            seq_type: SeqType::Protein,
        };
        let id = {
            let storage = Storage::open(temp.path()).unwrap();
            let id = storage.insert(&record).unwrap();
            storage.flush().unwrap();
            id
            // db closed
        };
        {
            // db reopened
            let storage = Storage::open(temp.path()).unwrap();
            let retrieved = storage.get(id).unwrap().unwrap();

            assert_eq!(record.header, retrieved.header);
            assert_eq!(record.sequence, retrieved.sequence);
        }
    }

}
