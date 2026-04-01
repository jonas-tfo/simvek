use ml_models::python_embedder::PythonEmbedder;
use core_db::types::{VectorDBConfig, VectorDB, FastaRecord, SeqType};
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let embedder = PythonEmbedder::new(
        &PathBuf::from("scripts/embed_query.py"),
        "Rostlab/prot_t5_xl_half_uniref50-enc",
        1024,  // ProtT5 output dim
    );

    let config = VectorDBConfig::default(PathBuf::from("data/db"));

    let mut db = VectorDB::open_fresh(config, Box::new(embedder as PythonEmbedder))?;

    // insert test records
    let test_seqs = [
        ("seq1",  b"MKTAYIAKQRQISFVKSHFSRQ" as &[u8]),
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
    for (header, seq) in &test_seqs {
        db.insert(FastaRecord {
            header: header.to_string(),
            sequence: seq.to_vec(),
            seq_type: SeqType::Protein,
        })?;
    }

    // test search
    let query = b"MKTAYIAKQRQISFVKSHFSRQ";
    let results = db.search(core_db::types::HnswSearchQuery {
        data: query,
        knn: 5,
        search_width: 50,
    })?;

    for neighbour in results {
        println!("{} — distance: {:.4}", neighbour.get_origin_id(), neighbour.get_distance());
    }

    Ok(())
}
