import argparse
from rankify.indexing.bm25_indexer import BM25Indexer

def main():
    print("CLI Started")
    parser = argparse.ArgumentParser(description="Rankify Indexer CLI")
    subparsers = parser.add_subparsers(dest="command")

    index_parser = subparsers.add_parser("index", help="Index a corpus")
    index_parser.add_argument("corpus_path", type=str, help="Path to the corpus JSONL file")
    index_parser.add_argument("--retriever", type=str, choices=["bm25"], default="bm25", help="Retriever to use")
    index_parser.add_argument("--output", default="rankify_indices", help="Output directory for index")
    index_parser.add_argument("--chunk_size", type=int, default=1024, help="Lines per chunk")
    index_parser.add_argument("--threads", type=int, default=32, help="Thread count for processing")
    index_parser.add_argument("--index_type", type=str, default="wiki", help="Type of index to create (default: wiki)")

    args = parser.parse_args()

    if args.command == "index":
        if args.retriever == "bm25":
            indexer = BM25Indexer(corpus_path=args.corpus_path, output_dir=args.output,
                                chunk_size=args.chunk_size, threads=args.threads, index_type=args.index_type)
            indexer.build_index()
            indexer.load_index()
            print("BM25 indexing complete.")


