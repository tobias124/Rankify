import logging
from rankify.retrievers.utils import process_chunk, generate_chunks, count_file_lines
from multiprocessing.dummy import Pool
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

def to_pyserini_jsonl(input_path, output_dir, chunk_size, threads) -> str:
    """Convert a corpus file to Pyserini JSONL format."""
    output_path = output_dir / "pyserini_corpus.json"

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
    output_path = output_dir / "pyserini_dense_corpus.json"

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