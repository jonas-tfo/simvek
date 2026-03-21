use ml_models::python_embedder::PythonEmbedder;
use core_db::types::VectorDBConfig;
use core_db::types::VectorDB;
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let embedder = PythonEmbedder::new(
        &PathBuf::from("scripts/embed_query.py"),
        "Rostlab/prot_t5_xl_half_uniref50-enc",
        1024,  // ProtT5 output dim
    );

    let config = VectorDBConfig::default(PathBuf::from("data/db"));

    let db = VectorDB::open(config, Box::new(embedder as PythonEmbedder))?;

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
