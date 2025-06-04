import logging
import pickle
from pathlib import Path

import numpy as np
import torch

from rankify.retrievers.utils import process_chunk, generate_chunks, count_file_lines
from multiprocessing.dummy import Pool
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import json
logging.basicConfig(level=logging.INFO)

def to_pyserini_jsonl(input_path, output_dir, chunk_size, threads) -> str:
    """Convert a corpus file to Pyserini JSONL format."""
    output_path = output_dir / "corpus.jsonl"

    total_lines = count_file_lines(input_path)
    logging.info("Streaming and processing corpus with " + str(threads) + " threads...")

    with open(input_path, "r", encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out, \
         Pool(processes=threads) as pool:

        pbar = tqdm(total=total_lines, desc="Processing")

        for result in pool.imap_unordered(lambda args: process_chunk(*args), generate_chunks(f_in, chunk_size)):
            for doc_json in result:
                f_out.write(doc_json + "\n")
            pbar.update(len(result))

        pbar.close()

    logging.info(f"✅ Done saving corpus to {output_path}")
    return output_path

def to_pyserini_jsonl_dense(input_path, output_dir, chunk_size, threads):
    """Convert a dense corpus file to Pyserini JSONL format."""
    output_path = output_dir / "corpus.jsonl"

    total_lines = count_file_lines(input_path)
    logging.info("Streaming and processing dense corpus with " + str(threads) + " threads...")

    with open(input_path, "r", encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out, \
         Pool(processes=threads) as pool:

        pbar = tqdm(total=total_lines, desc="Processing")

        for result in pool.imap_unordered(lambda args: process_chunk(*args, dense_index=True), generate_chunks(f_in, chunk_size)):
            for doc_json in result:
                f_out.write(doc_json + "\n")
            pbar.update(len(result))

        pbar.close()

    logging.info(f"Done saving dense corpus to {output_path}")
    return output_path

def to_contriever_embedding_chunks(jsonl_path, output_dir, chunk_size, model_name, batch_size, device):
    """
    Converts a corpus to Contriever embeddings in .pkl chunks.

    Returns:
        List[Path]: Paths to all .pkl shards.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    shard_paths = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        chunk = []
        shard_id = 0

        for idx, line in enumerate(f):
            doc = json.loads(line.strip())
            text = doc.get("contents", doc.get("text", "")).strip()
            if not text:
                continue
            doc_id = doc.get("id", f"doc{idx}")
            chunk.append((doc_id, text))

            if len(chunk) >= chunk_size:
                ids, embeddings = encode_contriever_chunk(chunk, tokenizer, model, batch_size, device)
                path = output_dir / f"embeddings_{shard_id}.pkl"
                with open(path, "wb") as fout:
                    pickle.dump((ids, embeddings), fout)
                shard_paths.append(path)
                shard_id += 1
                chunk = []

        if chunk:
            ids, embeddings = encode_contriever_chunk(chunk, tokenizer, model, batch_size, device)
            path = output_dir / f"embeddings_{shard_id}.pkl"
            with open(path, "wb") as fout:
                pickle.dump((ids, embeddings), fout)
            shard_paths.append(path)

    return shard_paths


def encode_contriever_chunk(chunk, tokenizer, model, batch_size, device):
    """
    Encodes a chunk of text using Contriever.
    """
    ids = []
    all_embeddings = []

    for i in tqdm(range(0, len(chunk), batch_size), desc="Encoding chunk"):
        batch = chunk[i:i+batch_size]
        batch_ids = [doc_id for doc_id, _ in batch]
        texts = [text for _, text in batch]

        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()

        all_embeddings.append(embeddings)
        ids.extend(batch_ids)

    embeddings = np.vstack(all_embeddings)
    return ids, embeddings
