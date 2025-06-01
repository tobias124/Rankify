# sample indexing code
from rankify.indexing import BM25Indexer

def index_bm25():
    """
    Sample code to index a BM25 index using the BM25Indexer class.
    This function initializes the indexer with the corpus path and output directory,
    builds the index, and loads it.
    """
    indexer = BM25Indexer(corpus_path="./data/wikipedia_100.jsonl", output_dir='rankify_indices',
                          chunk_size=1024, threads=8, index_type="wiki")

    indexer.build_index()

    indexer.load_index()

    print("BM25 indexing complete.")