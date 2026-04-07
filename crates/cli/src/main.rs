use clap::{Parser, Subcommand};
use std::{path::PathBuf, time::Duration};
use core_db::{HnswSearchQuery, SeqType, VectorDB, VectorDBConfig, types::str2seqtype};
use ml_models::python_embedder::PythonEmbedder;
use indicatif::{ProgressBar, ProgressStyle};

#[derive(Subcommand)]
enum Command {
    Build {
        /// fasta to build from
        #[arg(short, long)]
        fasta: String,
        /// path to the database
        #[arg(short, long, default_value = "data/db")]
        db_path: String,
        /// sequence type
        #[arg(short, long, default_value = "protein", value_parser = ["dna", "rna", "protein"])]
        record_type: String,
        /// wipe existing before building
        #[arg(long, default_value_t = false)]
        fresh: bool,
        /// model used to generate embeddings (name as given on huggingface, e.g. Rostlab/prot_bert)
        #[arg(short, long, default_value = "Rostlab/prot_bert")]
        model: String,
        /// the dimension of the output layer for the given model
        #[arg(short = 'n', long, default_value_t = 1024)]
        dim: u32,
    },
    Query {
        /// query sequence
        #[arg(short, long)]
        query: String,
        /// sequence type
        #[arg(short, long, default_value = "protein", value_parser = ["dna", "rna", "protein"])]
        record_type: String,
        /// number of nearest neighbours to return
        #[arg(short = 'k', long, default_value_t = 5)]
        top_k: usize,
        /// path to the database
        #[arg(short, long, default_value = "data/db")]
        db_path: String,
        /// output format: txt, json
        #[arg(short, long, default_value = "plain", value_parser = ["plain", "json"])]
        output: String,
        /// hnsw ef_construction: higher => more accurate index
        #[arg(long, default_value_t = 200)]
        ef_construction: usize,
        /// hnsw ef_search: query accuracy vs speed
        #[arg(long, default_value_t = 50)]
        ef_search: usize,
        /// model used to generate embeddings (name as given on huggingface, e.g. Rostlab/prot_bert)
        #[arg(short, long, default_value = "Rostlab/prot_bert")]
        model: String,
        /// the dimension of the output layer for the given model
        #[arg(short = 'n', long, default_value_t = 1024)]
        dim: u32,
    },
}

  #[derive(Parser)]
  struct Args {
      #[command(subcommand)]
      command: Command,
  }

fn main() {
    let args = Args::parse();
    match args.command {
        Command::Build { fasta, db_path, record_type, fresh, model, dim } => {
            let seq_type = str2seqtype(&record_type).unwrap();
            let conf = VectorDBConfig::default(PathBuf::from(&db_path), seq_type);
            let embedder = PythonEmbedder::new(
                &PathBuf::from("scripts/embed_query.py"),
                &model,
                dim as usize
            );
            let mut db = match fresh {
                true => VectorDB::open_fresh(conf, Box::new(embedder)).unwrap(),
                false => VectorDB::open(conf, Box::new(embedder)).unwrap(),
            };
            let fast = PathBuf::from(&fasta);
            let spinner = ProgressBar::new_spinner();
            spinner.set_style(ProgressStyle::default_spinner().template("{spinner:.green} {msg}").unwrap());
            spinner.set_message(format!("Embedding sequences from {} ...", fasta));
            spinner.enable_steady_tick(Duration::from_millis(100));
            VectorDB::rebuild_index_from_fasta_batch(&mut db, &fast).unwrap();
            spinner.finish_with_message("Done.");
        },
        Command::Query { query, record_type, top_k, db_path, output, ef_construction, ef_search, model, dim } => {
            let seq_type = str2seqtype(&record_type).unwrap();
            let conf = VectorDBConfig {
                path: PathBuf::from(db_path),
                ef_construction: ef_construction,
                ef_search,
                max_nb_connection: 16,
                expected_size: 100_000,
                max_layers: 16,
                record_type: seq_type
            };
            let embedder = Box::new(PythonEmbedder::new(
                &PathBuf::from("scripts/embed_query.py"),
                &model,
                dim as usize
            ));
            let spinner = ProgressBar::new_spinner();
            spinner.set_style(ProgressStyle::default_spinner().template("{spinner:.green} {msg}").unwrap());
            spinner.set_message(format!("Embedding sled database sequences for rebuilding of vector data base..."));
            spinner.enable_steady_tick(Duration::from_millis(100));
            let db = VectorDB::open(conf, embedder).unwrap();
            let query_bytes = query.into_bytes();
            let search_query = HnswSearchQuery {
                data: &query_bytes,
                knn: top_k,
                search_width: ef_search
            };
            spinner.set_message(format!("Embedding query and getting {} nearest neighbours", top_k));
            let neighbours = db.search(search_query).unwrap();
            spinner.finish_with_message("Done");
            match output.as_str() {
                "plain" => {
                    for n in neighbours {
                        let id: u64 = n.get_origin_id() as u64;
                        println!("id: {}, distance: {:.4}", n.d_id, n.distance);
                        let sled_sequence = db.sled_storage.get(id, SeqType::Protein).unwrap();
                        match sled_sequence {
                            Some(record) => {
                              println!(
                                  "Internal ID: {} -- distance: {:.4} -- seq: {}",
                                  n.get_origin_id(),
                                  n.get_distance(),
                                  String::from_utf8_lossy(&record.sequence)
                              );
                            }
                            _ => println!("Could not find sequence in sled, {} -- distance: {:.4}", id, n.get_distance())
                        };

                    }
                },
                "json" => {
                  let output: Vec<serde_json::Value> = neighbours.iter()
                      .map(|n| {
                          let id = n.get_origin_id() as u64;
                          let seq = db.sled_storage.get(id, SeqType::Protein).unwrap()
                              .map(|r| String::from_utf8_lossy(&r.sequence).into_owned());
                          serde_json::json!({
                              "id": n.d_id,
                              "distance": n.distance,
                              "sequence": seq,
                          })
                      })
                      .collect();
                  println!("{}", serde_json::to_string_pretty(&output).unwrap());
                },
                _ => unreachable!()
            }

        }
    }

}
