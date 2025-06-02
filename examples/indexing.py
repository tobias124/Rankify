# sample indexing code
from rankify.indexing import LuceneIndexer, DPRIndexer


def index_bm25():
    """
    Sample code to index a BM25 index using the BM25Indexer class.
    This function initializes the indexer with the corpus path and output directory,
    builds the index, and loads it.
    """
    indexer = LuceneIndexer(corpus_path="./data/wikipedia_100.jsonl", output_dir='rankify_indices',
                            chunk_size=1024, threads=8, index_type="wiki")

    indexer.build_index()

    indexer.load_index()

    print("BM25 indexing complete.")

def index_dpr_wiki():
        indexer = LuceneIndexer(corpus_path="./data/wikipedia_100.jsonl", output_dir="rankify_indices",
                                chunk_size=1024, threads=8, index_type="wiki", retriever_name="dpr")
        indexer.build_index()
        indexer.load_index()
        print("DPR wiki indexing complete.")

def index_dpr_msmarco():
        indexer = DPRIndexer(
            corpus_path="data/msmarco_100.jsonl",output_dir="rankify_indices",
            index_type="msmarco",
            encoder_name="facebook/dpr-ctx_encoder-single-nq-base",
            batch_size=32
        )

        indexer.build_index()
        indexer.load_index()

        print("DPR msmarco indexing complete.")