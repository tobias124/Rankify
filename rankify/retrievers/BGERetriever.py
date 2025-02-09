import os
import shutil
import requests
import h5py
import pickle
import faiss
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from rankify.dataset.dataset import Document, Context
from pyserini.eval.evaluate_dpr_retrieval import has_answers, SimpleTokenizer
from typing import List
import subprocess
from rankify.utils.pre_defind_models import INDEX_TYPE
import zipfile

import os
import zipfile
import requests
from tqdm import tqdm



class MultiPartZipExtractor:
    INDEX_PART1_URL = "https://huggingface.co/datasets/abdoelsayed/reranking-datasets/resolve/main/index/bgb_index.z01?download=true"
    INDEX_PART2_URL = "https://huggingface.co/datasets/abdoelsayed/reranking-datasets/resolve/main/index/bgb_index.zip?download=true"
    CACHE_DIR = os.path.join(os.path.expanduser('~'),'.cache','rankify', 'index') #os.environ.get("RERANKING_CACHE_DIR", "./cache")
    INDEX_FOLDER = os.path.join(CACHE_DIR, "bgb_index_wiki")

    def __init__(self):
        os.makedirs(self.INDEX_FOLDER, exist_ok=True)

    def download_file(self, url, save_path):
        """Download a file from a URL to a specified path."""
        if not os.path.exists(save_path):
            print(f"Downloading {os.path.basename(save_path)}...")
            response = requests.get(url, stream=True)
            with open(save_path, "wb") as f:
                for chunk in tqdm(response.iter_content(chunk_size=1024), desc=f"Downloading {os.path.basename(save_path)}"):
                    f.write(chunk)

    def extract_with_7z(self, zip_path, extract_to):
        """Extract a multi-part ZIP file using 7z."""
        print(f"Extracting archive {os.path.basename(zip_path)} with 7z...")
        try:
            print(zip_path)
            subprocess.run(
                ["7z", "x", zip_path, f"-o{extract_to}"],
                check=True
            )
            print("Extraction complete.")
        except subprocess.CalledProcessError as e:
            print(f"Extraction failed: {e}")
            raise RuntimeError("Failed to extract the combined ZIP archive with 7z.")
    def clean_up(self, *paths):
        """Remove specified files or directories."""
        for path in paths:
            if os.path.exists(path):
                os.remove(path)
                print(f"Removed file: {path}")
    def ensure_index_extracted(self):
        """Ensure that the index is downloaded, combined, and extracted."""
        part1_path = os.path.join(self.INDEX_FOLDER, "bgb_index.z01")
        part2_path = os.path.join(self.INDEX_FOLDER, "bgb_index.zip")

        # Step 1: Download parts
        if not os.path.exists(part1_path):
            self.download_file(self.INDEX_PART1_URL, part1_path)
        if not os.path.exists(part2_path):
            self.download_file(self.INDEX_PART2_URL, part2_path)

        # Step 2: Use 7z to extract directly from multi-part ZIP
        try:
            self.extract_with_7z(part2_path, self.INDEX_FOLDER)
        except Exception as e:
            print(f"Extraction failed: {e}")
            raise RuntimeError("Failed to extract combined archive.")
        
        #self.clean_up(part1_path, part2_path)

class BGERetriever:
    """
    BGERetriever: A Passage Retrieval System Using Precomputed Embeddings and FAISS Index `[1]`_.

    .. _[1]: https://dl.acm.org/doi/10.1145/3625555.3662062

    This retriever utilizes **precomputed FAISS indexes** to efficiently retrieve relevant documents based on query embeddings. 
    It supports **multi-part zip extraction** and **automatic passage retrieval** for large-scale document retrieval tasks.

    References
    ----------
    .. [1] Xiao et al. (2024): C-Pack: Packed Resources for General Chinese Embeddings.

    Attributes
    ----------
    model_name : str
        The embedding model used for generating dense representations.
    n_docs : int
        The number of top documents to retrieve for each query.
    batch_size : int
        Batch size for encoding queries in parallel.
    device : str
        Device used for encoding (`cuda` or `cpu`).
    index_type : str
        Type of index to use (`wiki` or `msmarco`).
    index_folder : str
        Path where the FAISS index and supporting files are stored.
    doc_ids : List[str]
        List of document IDs mapped to stored embeddings.
    doc_texts : dict
        Dictionary mapping document IDs to text and title.
    passage_path : str
        Path to the downloaded passage file.
    index : faiss.IndexFlatIP
        The FAISS index used for nearest neighbor search.
    model : AutoModel
        The transformer model used for query encoding.
    tokenizer : AutoTokenizer
        The tokenizer corresponding to the embedding model.

    Examples
    --------
    Basic Usage:

    >>> from rankify.dataset.dataset import Document, Question
    >>> from rankify.retrievers.retriever import Retriever
    >>> retriever = Retriever( method = "bge",model="BAAI/bge-large-en-v1.5", n_docs=5, index_type="wiki")
    >>> documents = [Document(question=Question("Who discovered gravity?"))]
    >>> retrieved_documents = retriever.retrieve(documents)
    >>> print(retrieved_documents[0].contexts[0].text)
    """

    CACHE_DIR = os.environ.get("RERANKING_CACHE_DIR", "./cache")

    def __init__(self, model="BAAI/bge-large-en-v1.5", n_docs=10, batch_size=32, device="cuda", index_type="wiki"):
        """
        Initializes the BGB retriever.

        Parameters
        ----------
        model_name : str
            The name of the embedding model.
        n_docs : int
            Number of documents to retrieve per query.
        batch_size : int
            Batch size for encoding queries.
        device : str
            The device to use for encoding ('cuda' or 'cpu').
        index_type : str
            The type of index to use ('wiki' or 'msmarco').
        """
        self.model_name = model
        self.n_docs = n_docs
        self.batch_size = batch_size
        self.device = device
        self.index_type = index_type

        # Get configuration for the specified index type
        if 'bge' not in INDEX_TYPE or index_type not in INDEX_TYPE['bge']:
            raise ValueError(f"Index type '{index_type}' is not supported for BGBRetriever.")

        self.index_config = INDEX_TYPE['bge'][index_type]
        self.index_folder = os.path.join(self.CACHE_DIR,  "index" ,f"bgb_index_{index_type}")
        self.passages_url = self.index_config.get("passages_url")
        self.passage_path = None

        self._ensure_index_and_passages_downloaded()
        self.doc_ids = self._load_document_ids()
        self.doc_texts = self._load_tsv()
        self.build_faiss_index_on_disk()

        # Load the model and tokenizer
        print(f"Loading model: {self.model_name}")
        self.model_hf = AutoModel.from_pretrained(self.model_name).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def _ensure_index_and_passages_downloaded(self):
        """
        Ensures that the necessary index files and passages are downloaded and extracted.
        """
        os.makedirs(self.index_folder, exist_ok=True)

        # Handle 'wiki' index
        if self.index_type == "wiki":
            embeddings_path = os.path.join(self.index_folder, "bgb_index", "bge_embeddings.h5")
            doc_ids_path = os.path.join(self.index_folder, "bgb_index", "bge_doc_ids.pkl")
            #print(embeddings_path)
            #asdasda
            # Check if required files exist
            if not os.path.exists(embeddings_path) or not os.path.exists(doc_ids_path):
                print("Wiki embeddings or document IDs missing. Proceeding to download and extract...")
                extractor = MultiPartZipExtractor()
                extractor.ensure_index_extracted()
            else:
                print("Wiki embeddings and document IDs already exist. Skipping download.")

        # Handle 'msmarco' index
        elif self.index_type == "msmarco":
            embeddings_path = os.path.join(self.index_folder, "bgb_index", "bge_embeddings.h5")
            doc_ids_path = os.path.join(self.index_folder, "bgb_index", "bge_doc_ids.pkl")
            index_zip_path = os.path.join(self.index_folder, "index.zip")

            if not os.path.exists(embeddings_path) or not os.path.exists(doc_ids_path):
                print("MSMARCO index files missing. Proceeding to download and extract...")
                self._download_file(self.index_config["url"], index_zip_path)

                with zipfile.ZipFile(index_zip_path, "r") as zip_ref:
                    zip_ref.extractall(self.index_folder)

                # Cleanup to save space
                os.remove(index_zip_path)
            else:
                print("MSMARCO index files already exist. Skipping download.")


        # Handle passages download
        if self.passages_url:
            passage_file_name = self.clean_filename(self.passages_url)
            passage_path = os.path.join(self.CACHE_DIR, passage_file_name)
            if not os.path.exists(passage_path):
                print(f"Downloading passages '{passage_file_name}' into main cache...")
                response = requests.get(self.passages_url, stream=True)
                with open(passage_path, "wb") as f:
                    for chunk in tqdm(response.iter_content(chunk_size=1024), desc=f"Downloading Passages ({passage_file_name})"):
                        f.write(chunk)
            else:
                print(f"Passages '{passage_file_name}' already exist in the main cache.")
            self.passage_path = passage_path


    def _download_file(self, url, save_path):
        """
        Downloads a file from the specified URL to the given path.
        """
        print(f"Downloading {os.path.basename(save_path)}...")
        response = requests.get(url, stream=True)
        with open(save_path, "wb") as f:
            for chunk in tqdm(response.iter_content(chunk_size=1024), desc=f"Downloading {os.path.basename(save_path)}"):
                f.write(chunk)

    def clean_filename(self, url):
        """
        Cleans the filename from a URL by removing query parameters.
        """
        return url.split("/")[-1].split("?")[0]

    def build_faiss_index_on_disk(self):
        """
        Builds or loads a FAISS index depending on the index type.
        """
        print("Handling FAISS index...")

        index_path = os.path.join(self.index_folder, "bgb_index", "faiss_index.bin")
        embeddings_path = os.path.join(self.index_folder, "bgb_index", "bge_embeddings.h5")

        if os.path.exists(index_path):
            print(f"Loading existing FAISS index for '{self.index_type}' from {index_path}...")
            self.index = faiss.read_index(index_path)
        else:
            print(f"Building FAISS index for '{self.index_type}' from embeddings at {embeddings_path}...")
            with h5py.File(embeddings_path, "r") as f:
                embeddings = f["embeddings"][:].astype(np.float32)

            embedding_dim = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(embedding_dim)

            chunk_size = 50000
            for start in tqdm(range(0, embeddings.shape[0], chunk_size), desc="Adding embeddings to FAISS index"):
                end = min(start + chunk_size, embeddings.shape[0])
                chunk = embeddings[start:end]
                self.index.add(chunk)

            print(f"Saving FAISS index for '{self.index_type}' to {index_path}...")
            faiss.write_index(self.index, index_path)

        print(f"FAISS index loaded with {self.index.ntotal} embeddings.")

    def _load_document_ids(self):
        """
        Loads the document IDs for the respective index type.

        Returns
        -------
        list
            The list of document IDs.
        """
        doc_ids_path = os.path.join(self.index_folder, "bgb_index", "bge_doc_ids.pkl")
        print(f"Loading document IDs for '{self.index_type}' from {doc_ids_path}...")
    
        with open(doc_ids_path, "rb") as f:
                doc_ids = []
                while True:
                    try:
                        doc_ids.extend(pickle.load(f))
                    except EOFError:
                        break

        print(f"Loaded {len(doc_ids)} document IDs for '{self.index_type}'.")
        return doc_ids


    def _load_tsv(self):
        """
        Loads the TSV file and maps document IDs to their text and title.

        Returns
        -------
        dict
            A dictionary mapping document IDs to text and title.
        """
        print(f"Loading passages from {self.passage_path}...")
        doc_texts = {}
        with open(self.passage_path, "r", encoding="utf-8") as f:
            next(f)  # Skip header
            for line in f:
                doc_id, passage, title = line.strip().split("\t")
                doc_texts[doc_id] = {"text": passage, "title": title}
        return doc_texts

    def _encode_queries(self, queries):
        """
        Encodes queries using the embedding model.

        Parameters
        ----------
        queries : list of str
            List of query strings.

        Returns
        -------
        np.ndarray
            The query embeddings.
        """
        all_embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0, len(queries), self.batch_size), desc="Encoding queries"):
                batch = queries[i:i + self.batch_size]
                tokenized = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(self.device)
                model_output = self.model_hf(**tokenized)
                embeddings = model_output.last_hidden_state[:, 0, :]  # CLS token
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                all_embeddings.append(embeddings.cpu().numpy())
        return np.vstack(all_embeddings)

    def retrieve(self, documents: List[Document]) -> List[Document]:
        """
        Retrieve contexts for each document in the input list using the document's question.

        Parameters
        ----------
        documents : List[Document]
            A list of Document objects containing the queries and answers.

        Returns
        -------
        List[Document]
            The list of Document objects updated with retrieved contexts.
        """
        queries = [doc.question.question for doc in documents]
        query_embeddings = self._encode_queries(queries)

        # Debugging: Log dimensionality
        #print(f"Query embeddings dimensionality: {query_embeddings.shape[1]}")
        #print(f"FAISS index dimensionality: {self.index.d}")

        # Ensure dimensions match
        if query_embeddings.shape[1] != self.index.d:
            raise ValueError(
                f"Dimensionality mismatch: Query embeddings have dimension {query_embeddings.shape[1]}, "
                f"but FAISS index has dimension {self.index.d}."
            )


        
        # Initialize empty results for batch processing
        all_distances = []
        all_indices = []

        # Batch processing for FAISS search
        for start_idx in tqdm(range(0, len(query_embeddings), self.batch_size), desc="Batch FAISS Search"):
            end_idx = min(start_idx + self.batch_size, len(query_embeddings))
            batch_query_embeddings = query_embeddings[start_idx:end_idx]

            distances, indices = self.index.search(batch_query_embeddings, self.n_docs)
            all_distances.append(distances)
            all_indices.append(indices)

        # Concatenate all batch results
        all_distances = np.vstack(all_distances)
        all_indices = np.vstack(all_indices)

        # Process the results for each document
        for i, document in enumerate(tqdm(documents, desc="Processing documents", unit="doc")):
            contexts = []
            for dist, idx in zip(all_distances[i], all_indices[i]):
                doc_id = self.doc_ids[idx]
                doc_text = self.doc_texts.get(str(doc_id), {"text": "Text not found", "title": "No Title"})
                context = Context(
                    id=doc_id,
                    title=doc_text["title"],
                    text=doc_text["text"],
                    score=dist,
                    has_answer=has_answers(doc_text["text"], document.answers.answers, SimpleTokenizer())
                )
                contexts.append(context)
            document.contexts = contexts

        return documents
    

