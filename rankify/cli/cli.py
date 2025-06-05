import argparse

from rankify.indexing.colbert_indexer import ColBERTIndexer
from rankify.indexing.lucene_indexer import LuceneIndexer
from rankify.indexing.dpr_indexer import DPRIndexer
from rankify.indexing.contriever_indexer import ContrieverIndexer
from pathlib import Path

SUPPORTED_RETRIEVERS = ["bm25", "dpr", "contriever", "colbert"]

def handle_output_directory(output_dir):
    """
    Ensure the output directory exists, creating it if necessary.
    """
    if not output_dir:
        print("Output directory is not specified. Using default 'rankify_indices'.")
        output_dir = "rankify_indices"
    else:
        output_path = Path(output_dir)
        if not output_path.exists():
            print(f"Output directory {output_dir} does not exist. Creating it.")
            output_path.mkdir(parents=True, exist_ok=True)
    return output_dir

def main():
    print("CLI Started")
    parser = argparse.ArgumentParser(description="Rankify Indexer CLI")
    subparsers = parser.add_subparsers(dest="command")

    index_parser = subparsers.add_parser("index", help="Index a corpus")
    index_parser.add_argument("corpus_path", type=str, help="Path to the corpus JSONL file")
    index_parser.add_argument("--retriever", type=str, choices=SUPPORTED_RETRIEVERS, help="Retriever to use")
    index_parser.add_argument("--output", default="rankify_indices", help="Output directory for index")
    index_parser.add_argument("--chunk_size", type=int, default=1024, help="Lines per chunk")
    index_parser.add_argument("--threads", type=int, default=32, help="Thread count for processing")
    index_parser.add_argument("--index_type", type=str, default="wiki", help="Type of index to create (default: wiki)")

    # Dense-specific option
    #Todo: use encoder map - to input only model name
    index_parser.add_argument("--encoder", type=str, help="Encoder model name for dense indexing")
    index_parser.add_argument("--batch_size", type=int, default=32, help="Batch size for DPR encoding")
    index_parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cuda", help="Device to use for indexing (default: cuda)")

    args = parser.parse_args()

    if args.command == "index":
        handle_output_directory(args.output)
        args.retriever = args.retriever.lower()

        if args.retriever == "bm25":
            indexer = LuceneIndexer(corpus_path=args.corpus_path, output_dir=args.output,
                                    chunk_size=args.chunk_size, threads=args.threads, index_type=args.index_type)
            indexer.build_index()
            indexer.load_index()
            print("BM25 indexing complete.")

        elif args.retriever == "dpr":
            if args.index_type == "wiki":
                indexer = LuceneIndexer(corpus_path=args.corpus_path, output_dir=args.output,
                                        chunk_size=args.chunk_size, threads=args.threads, index_type=args.index_type,
                                        retriever_name="dpr")
                indexer.build_index()
                indexer.load_index()
                print("DPR wiki indexing complete.")
                return
            else:
                if args.encoder is None:
                    args.encoder = "facebook/dpr-ctx_encoder-single-nq-base"
                    print("No encoder specified. Using default: facebook/dpr-ctx_encoder-single-nq-base")

                indexer = DPRIndexer(
                    corpus_path=args.corpus_path,
                    output_dir=args.output,
                    chunk_size=args.chunk_size,
                    threads=args.threads,
                    index_type=args.index_type,
                    encoder_name=args.encoder,
                    batch_size=args.batch_size,
                    device=args.device
                )

                indexer.build_index()
                indexer.load_index()

                print("DPR indexing complete.")

        elif args.retriever == "contriever":

            if args.encoder is None:
                #Todo: check if supported
                args.encoder = "facebook/contriever"
                print("No encoder specified. Using default: facebook/contriever")

            indexer = ContrieverIndexer(
                corpus_path=args.corpus_path,
                output_dir=args.output,
                chunk_size=args.chunk_size,
                threads=args.threads,
                index_type=args.index_type,
                encoder_name=args.encoder,
                batch_size=args.batch_size,
                device=args.device
            )

            indexer.build_index()
            indexer.load_index()

            print("Contriever indexing complete.")

        elif args.retriever == "colbert":
            if args.encoder is None:
                #Todo: check if supported
                args.encoder = "colbert-ir/colbertv2.0"
                print("No encoder specified. Using default: colbert-ir/colbertv2.0")

            indexer = ColBERTIndexer(
                corpus_path=args.corpus_path,
                output_dir=args.output,
                chunk_size=args.chunk_size,
                threads=args.threads,
                index_type=args.index_type,
                encoder_name=args.encoder,
                batch_size=args.batch_size,
                device=args.device
            )

            indexer.build_index()
            indexer.load_index()

            print("ColBERT indexing complete.")
        else:
            print(f"Unknown retriever type: {args.retriever}. Supported types are {SUPPORTED_RETRIEVERS}.")
