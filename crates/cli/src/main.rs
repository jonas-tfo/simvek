use clap::{Parser, Subcommand};
use std::{path::PathBuf, time::Duration};
use core_db::{HnswDB, HnswDBConfig, HnswSearchQuery, SeqType, SequenceEmbedder, types::{str2seqtype, seqtype2str, FastaRecord}, fileutils::parse_fasta};
use ml_models::python_embedder::PythonEmbedder;
use indicatif::{ProgressBar, ProgressStyle};

#[derive(Subcommand)]
enum Command {
    Build {
        /// fasta to build from
        #[arg(short, long)]
        fasta: String,
        /// path to the sequence database
        #[arg(short, long, default_value = "data/sequence_db")]
        sequence_db_path: String,
        /// path to the vector database
        #[arg(short, long, default_value = "data/vector_db")]
        vector_db_path: String,
        /// sequence type
        #[arg(short, long, value_parser = ["dna", "rna", "protein"])]
        record_type: Option<String>,
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
        #[arg(short, long, value_parser = ["dna", "rna", "protein"])]
        record_type: Option<String>,
        /// number of nearest neighbours to return
        #[arg(short = 'k', long, default_value_t = 5)]
        top_k: usize,
        /// path to the sequence database
        #[arg(short, long, default_value = "data/sequence_db")]
        sequence_db_path: String,
        /// path to the vector database
        #[arg(short, long, default_value = "data/vector_db")]
        vector_db_path: String,
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
        Command::Build { fasta, sequence_db_path, vector_db_path, record_type, fresh, model, dim } => {
            let fasta_path = PathBuf::from(&fasta);

            // [1/3] parse
            let (ids, seqs, seq_type) = parse_fasta(&fasta_path).unwrap();
            let n = ids.len();
            println!("[1/3] Parsed {} sequences from {}", n, fasta);

            let final_seqtype: SeqType;

            match record_type {
                Some(t) => {
                    let record_type_seqtype = str2seqtype(&t).unwrap();
                    if record_type_seqtype != seq_type {
                        eprintln!("Warning: sequence type found in the fasta is not the same as the passed sequence type, the passed sequence type will be used instead")
                    }
                    final_seqtype = record_type_seqtype;
                },
                None => {
                    final_seqtype = seq_type;
                    eprintln!("Warning: No sequence type was passed, defaulting to {}", seqtype2str(final_seqtype).unwrap())
                }
            }

            // [2/3] embed
            let embedder = PythonEmbedder::new(&PathBuf::from("scripts/embed_query.py"), &model, dim as usize);
            let spinner = ProgressBar::new_spinner();
            spinner.set_style(ProgressStyle::default_spinner().template("{spinner:.green} {msg}").unwrap());
            spinner.set_message(format!("[2/3] Embedding {} sequences...", n));
            spinner.enable_steady_tick(Duration::from_millis(100));
            let embeddings = embedder.embed_batch(&fasta_path).unwrap();
            spinner.finish_with_message(format!("[2/3] Embedded {} sequences", n));

            // [3/3] index
            let conf = HnswDBConfig::default(PathBuf::from(&sequence_db_path), PathBuf::from(&vector_db_path), final_seqtype);
            let mut db = match fresh {
                true => HnswDB::open_fresh(conf, Box::new(embedder)).unwrap(),
                false => HnswDB::open(conf, Box::new(embedder)).unwrap(),
            };
            let pb = ProgressBar::new(n as u64);
            pb.set_style(ProgressStyle::default_bar()
                .template("[3/3] Indexing sequences and embeddings [{bar:40.cyan/blue}] {pos}/{len}").unwrap()
                .progress_chars("=> "));
            for ((id, seq), embedding) in ids.into_iter().zip(seqs).zip(embeddings) {
                let seq_bytes: Vec<u8> = seq.into_bytes();
                let record = FastaRecord { header: id, sequence: seq_bytes, seq_type };
                db.insert_with_embedding(record, embedding).unwrap();
                pb.inc(1);
            }
            pb.finish_with_message(format!("[3/3] Indexed {} sequences", n));
            println!("Sequence database written to {}", sequence_db_path);
            println!("Vector database written to {}", vector_db_path);
        },
        Command::Query { query, record_type, top_k, sequence_db_path, vector_db_path, output, ef_construction, ef_search, model, dim } => {
            let final_seqtype: SeqType;
            match record_type {
                Some(t) => {
                    final_seqtype = str2seqtype(&t).unwrap();
                },
                None => {
                    eprint!("Warning: no sequence type was set, defaulting to protein");
                    final_seqtype = SeqType::Protein;
                }

            }
            let conf = HnswDBConfig {
                sequence_sled_data_path: PathBuf::from(&sequence_db_path),
                vector_sled_data_path: PathBuf::from(&vector_db_path),
                ef_construction: ef_construction,
                ef_search,
                max_nb_connection: 16,
                expected_size: 100_000,
                max_layers: 16,
                record_type: final_seqtype
            };
            let embedder = Box::new(PythonEmbedder::new(
                &PathBuf::from("scripts/embed_query.py"),
                &model,
                dim as usize
            ));
            let spinner = ProgressBar::new_spinner();
            spinner.set_style(ProgressStyle::default_spinner().template("{spinner:.green} {msg}").unwrap());
            spinner.set_message("[1/2] Loading index from storage...");
            spinner.enable_steady_tick(Duration::from_millis(100));
            let db = HnswDB::open(conf, embedder).unwrap();
            let query_bytes = query.into_bytes();
            let search_query = HnswSearchQuery {
                data: &query_bytes,
                knn: top_k,
                search_width: ef_search
            };
            spinner.set_message(format!("[2/2] Embedding query and searching for {} nearest neighbours...", top_k));
            let neighbours = db.search(search_query).unwrap();
            spinner.finish();
            match output.as_str() {
                "plain" => {
                    for n in neighbours {
                        let id: u64 = n.get_origin_id() as u64;
                        println!("id: {}, distance: {:.4}", n.d_id, n.distance);
                        let sequence = db.sequence_db.get(id, final_seqtype).unwrap();
                        match sequence {
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
                            let seq = db.sequence_db.get(id, SeqType::Protein).unwrap()
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
