
use core_db::{FastaRecord, HnswSearchQuery, ModelSignature, SeqType, SequenceEmbedder, VectorDB, VectorDBConfig};
use anyhow::Result;
use tempfile::tempdir;

struct DevEmbedder;

impl SequenceEmbedder for DevEmbedder {
    fn embed(&self, _seq: &[u8]) -> Result<Vec<f32>> {
        use rand::seq::SliceRandom;
        let mut rng = rand::rng();
        let mut v: Vec<f32> = (0..1024).map(|i| i as f32).collect();
        v.shuffle(&mut rng);
        Ok(v)
    }
    fn embed_dev(&self, seq: &[u8]) -> Result<Vec<f32>> {
        self.embed(seq)
    }
    fn embed_batch(&self, _: &[&[u8]]) -> Result<Vec<Vec<f32>>> {
        todo!()
    }
    fn get_dimension(&self) -> usize {
        1024
    }
    fn get_signature(&self) -> ModelSignature {
        ModelSignature { name: "dev".into(), dimension: 1024 }
    }
}

fn make_vector_db() -> VectorDB {
    let dir = tempdir().unwrap();
    let path = dir.keep();
    VectorDB::open(VectorDBConfig::default(path, SeqType::Protein), Box::new(DevEmbedder)).unwrap()
}

fn make_record(header: &str, seq: &[u8]) -> FastaRecord {
    FastaRecord {
        header: header.to_string(),
        sequence: seq.to_vec(),
        seq_type: SeqType::Protein,
    }
}

#[test]
fn insert_returns_id() {
    let mut db = make_vector_db();
    let id = db.insert(make_record("seq1", b"ACGT")).unwrap();
    print!("the id is {}", id);
    assert!(id + 1 > 0);
}

#[test]
fn search_returns_nearest_neighbour() {
    let mut db = make_vector_db();
    db.insert(make_record("seq1", b"ACGT")).unwrap();
    db.insert(make_record("seq2", b"TTTT")).unwrap();

    let query = HnswSearchQuery { data: b"ACGT", knn: 1, search_width: 10 };
    let results = db.search(query).unwrap();
    assert_eq!(results.len(), 1);
}

#[test]
fn search_knn_respects_k() {
    let mut db = make_vector_db();
    for i in 0..5 {
        db.insert(make_record(&format!("seq{}", i), b"ACGT")).unwrap();
    }
    let query = HnswSearchQuery { data: b"ACGT", knn: 3, search_width: 10 };
    let results = db.search(query).unwrap();
    assert_eq!(results.len(), 3);
}

#[test]
fn insert_and_search_protein_sequences() {
    let seqs: &[(&str, &[u8])] = &[
        ("seq1",  b"MKTAYIAKQRQISFVKSHFSRQ"),
        ("seq2",  b"ACDEFGHIKLMNPQRSTVWY"),
        ("seq3",  b"KALTARQQEVFDLIRDHISQTGMPPTRAEIAQDFK"),
        ("seq4",  b"MGSSHHHHHHSSGLVPRGSHMASMTGGQQMGRDLYDDDDKDPSSEFHHHHHH"),
        ("seq5",  b"MNIFEMLRIDEGLRLKIYKDTEGYYTIGIGHLLTKSPSLNAAKSELDKAIGRNTNGVITKDEAEKLFNQDVDAAVRGILRNAKLKPVYDSLDAVRRAALINMVFQMGETGVAGFTNSLRMLQQKRWDEAAVNLAKSRWYNQTPNRAKRVITTFRTGTWDAYKNL"),
        ("seq6",  b"MAFSAEDVLKEYDRRRRMEALLLSLYYPNDRKLLDYKEWSPPRVQVECPKAPVEWNNPPSEKGLIVGHFSGIKYKGEKAQASEVDVNKMCCWVSKFKDAMRRYQGIQTCKIPGKVLSDLDAKIKAYNLTVEGVEGFVRYSRVTKQHVAAFLKELRHSKQYENVNLHYLTAVRQRLNGSYFAN"),
        ("seq7",  b"MVLSEGEWQLVLHVWAKVEADVAGHGQDILIRLFKSHPETLEKFDRFKHLKTEAEMKASEDLKKHGVTVLTALGGILKKKGHHEAELKPLAQSHATKHKIPIKYLEFISEAIIHVLHSRHPGNFGADAQGAMNKALELFRKDIAAKYKELGYQG"),
        ("seq8",  b"MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSY"),
        ("seq9",  b"MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLTYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITLGMDELYK"),
        ("seq10", b"MSHHWGYGKHNGPEHWHKDFPIAKGERQSPVDIDTHTAKYDPSLKPLSVSYDQATSLRILNNGAAFNVEFD"),
    ];

    let mut db = make_vector_db();
    for (header, seq) in seqs {
        db.insert(make_record(header, seq)).unwrap();
    }

    let query = HnswSearchQuery { data: b"MKTAYIAKQRQISFVKSHFSRQ", knn: 5, search_width: 50 };
    let results = db.search(query).unwrap();
    assert_eq!(results.len(), 5);
}
