use::anyhow;
use anyhow::{Error, Result};
use std::fs::File;
use std::io::{BufReader, BufRead};
use std::path::PathBuf;
use crate::SeqType;
use crate::constants::{is_dna, is_rna, is_prot};

pub fn parse_fasta(filename: &PathBuf) -> anyhow::Result<(Vec<String>, Vec<String>, SeqType)> {
    let file = File::open(filename)?;
    let reader = BufReader::new(file);
    let mut ids: Vec<String> = Vec::new();
    let mut sequences: Vec<String> = Vec::new();
    // pre allocate buffer to prevent re-allocation
    let mut current_sequence = String::with_capacity(1024);

    for line in reader.lines() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() { continue; }

        if line.starts_with('>') {
            if !(current_sequence.is_empty()) {
                sequences.push(current_sequence.to_string());
                current_sequence.clear(); // clear for new record
            }
            let id: &str = &line[1..];
            ids.push(id.to_string());
        } else {
            let seq_part = &line;
            current_sequence.push_str(seq_part);
        }
    }
    if !(current_sequence.is_empty()) {
        sequences.push(current_sequence.to_string());
    }
    let seq_type = sequences
        .first()
        .map(|seq| get_seq_type(seq))
        .unwrap_or(Ok(SeqType::Dna))?;
    return Ok((ids, sequences, seq_type))
}

fn get_seq_type(line: &str) -> Result<SeqType> {
    let mut maybe_dna = true;
    let mut maybe_rna = true;
    let mut maybe_prot = true;

    for ch in line.bytes() {
        if maybe_dna && !is_dna(ch) { maybe_dna = false; }
        if maybe_rna && !is_rna(ch) { maybe_rna = false; }
        if maybe_prot && !is_prot(ch) { maybe_prot = false; }
    }

    match (maybe_dna, maybe_rna, maybe_prot) {
        (true, _, _) => Ok(SeqType::Dna),
        (_, true, _) => Ok(SeqType::Rna),
        (_, _, true) => Ok(SeqType::Protein),
        _ => Err(Error::msg("Couldnt find seq type"))
    }
}
