import argparse
import sys
import json
import torch
from transformers import T5Tokenizer, T5EncoderModel

MODEL_NAME = "Rostlab/prot_t5_xl_half_uniref50-enc"

def embed_sequence(sequence: str, tokenizer, encoder, device: str) -> list[float]:
    # mean pool over residues to get a single vector per sequence
    seq_spaced = " ".join(list(sequence))

    tokenized = tokenizer(
        seq_spaced,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)

    with torch.no_grad():
        output = encoder(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = output.last_hidden_state  # (1, seq_len, hidden_dim)

    # strip EOS token
    seq_len = len(sequence)
    embeddings = embeddings.squeeze(0)[:seq_len]  # (seq_len, hidden_dim)

    # mean pool -> single vector for the whole sequence
    pooled = embeddings.mean(dim=0)  # (hidden_dim,)

    return pooled.float().cpu().tolist()


def main():
    parser = argparse.ArgumentParser(description="Embed a protein sequence with ProtT5")
    parser.add_argument("--sequence", required=True, help="amino acid sequence")
    parser.add_argument("--model", default=MODEL_NAME, help="huggingface model name")
    parser.add_argument("--device", default=None, help="cpu, cuda, or mps")
    args = parser.parse_args()

    if args.device is None:
        device = (
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )
    else:
        device = args.device

    tokenizer = T5Tokenizer.from_pretrained(args.model, do_lower_case=False)
    encoder = T5EncoderModel.from_pretrained(args.model, torch_dtype=torch.float16)
    encoder.to(device)
    encoder.eval()

    embedding = embed_sequence(args.sequence, tokenizer, encoder, device)

    # json to stdout for rust
    json.dump(embedding, sys.stdout)


if __name__ == "__main__":
    main()
