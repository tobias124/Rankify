import shutil
import subprocess
import logging


from rankify.indexing.base_indexer import BaseIndexer
from rankify.indexing.format_converters import to_pyserini_jsonl

logging.basicConfig(level=logging.INFO)

class BM25Indexer(BaseIndexer):
    """
    BM25 Indexer for creating and loading BM25 indices using Pyserini.
    This class handles the preparation of the corpus, building the index, and loading the index.
    """
    def __init__(self, corpus_path, output_dir="rankify_indices", chunk_size=1024, threads=32, index_type="wiki"):
        super().__init__(corpus_path, output_dir, chunk_size, threads, index_type)
        if index_type =="wiki":
            self.index_dir = self.output_dir / f"bm25_index"
        else:
            self.index_dir = self.output_dir / f"bm25_index_{index_type}"
    

    def _save_pyserini_corpus(self):
        """
        Convert the corpus to the Pyserini JSONL format.
        This method uses the `to_pyserini_jsonl` function to convert the corpus file
        :return:
        """
        return to_pyserini_jsonl(self.corpus_path, self.output_dir, self.chunk_size, self.threads)
       
    def build_index(self):
        """
        Build the BM25 index from the corpus.
        This method prepares the corpus in the format required by Pyserini,
        creates a temporary directory for the corpus, and runs the Pyserini indexer.
        It also handles the cleanup of temporary files and saves a title map for the indexed documents.
        :return:
        """
        corpus_path = self._save_pyserini_corpus()
        temp_corpus_dir = self.output_dir / "temp_corpus"

        if temp_corpus_dir.exists():
            shutil.rmtree(temp_corpus_dir)
        temp_corpus_dir.mkdir(parents=True)

        corpus_file_path = temp_corpus_dir / "corpus.json"
        corpus_path.rename(corpus_file_path)

        if self.index_dir.exists():
            shutil.rmtree(self.index_dir)
        self.index_dir.mkdir(parents=True)

        cmd = [
            "python", "-m", "pyserini.index.lucene",
            "-collection", "JsonCollection",
            "-generator", "DefaultLuceneDocumentGenerator",
            "-threads", str(self.threads),
            "-input", str(temp_corpus_dir),
            "-index", str(self.index_dir),
            "-storePositions", "-storeDocvectors", "-storeRaw"
        ]

        logging.info(f"Running Pyserini indexer:\n{' '.join(cmd)}")
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Pyserini indexing failed: {e}")

        shutil.rmtree(temp_corpus_dir)

        self.cleanup_lock_file(self.index_dir)
        self._save_title_map()

        logging.info(f"✅ Indexing complete. Index stored at {self.index_dir}")

    def load_index(self):
        """
        Load the BM25 index from the specified directory.

        This method checks if the index directory exists and is not empty,
        and raises an error if the index is locked or not found.
        :return:
        """
        if not self.index_dir.exists() or not any(self.index_dir.iterdir()):
            raise FileNotFoundError(f"Index directory {self.index_dir} does not exist or is empty.")
        lock_file = self.index_dir / ".lock"
        if lock_file.exists():
            raise RuntimeError(f"Index is locked: {lock_file}")
        logging.info(f"Index loaded from {self.index_dir}")

