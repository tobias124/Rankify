import os
import json
import requests
import zipfile
from typing import List
from pyserini.search.faiss import FaissSearcher
from pyserini.eval.evaluate_dpr_retrieval import has_answers, SimpleTokenizer
from rankify.dataset.dataset import Document, Context
from tqdm import tqdm
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("facebook/dpr-question_encoder-multiset-base")

class DenseRetriever:
    """
    Implements **Dense Passage Retrieval (DPR)** `[5]`_, **ANCE** `[6]`_, and **BPR** `[7]`_ using FAISS indexes 
    for efficient document retrieval. These models leverage dense vector representations to find semantically relevant 
    passages.

    .. _[5]: https://arxiv.org/abs/2004.04906
    .. _[6]: https://arxiv.org/abs/2007.00808
    .. _[7]: https://arxiv.org/abs/2106.00882

    References
    ----------
    .. [5] Karpukhin, V., et al. (2020). *Dense Passage Retrieval for Open-Domain Question Answering*.
           [https://arxiv.org/abs/2004.04906](https://arxiv.org/abs/2004.04906)
    .. [6] Xiong, L., et al. (2020). *Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval (ANCE)*.
           [https://arxiv.org/abs/2007.00808](https://arxiv.org/abs/2007.00808)
    .. [7] Ni, J., et al. (2021). *Efficient Passage Retrieval with Hashing for Open-domain Question Answering*.
           [https://arxiv.org/abs/2106.00882](https://arxiv.org/abs/2106.00882)

    Attributes
    ----------
    model : str
        The type of retriever (`"dpr-multi"`, `"dpr-single"`, `"ance-multi"`, `"bpr-single"`).
    index_type : str
        Dataset type (`"wiki"` or `"msmarco"`).
    n_docs : int
        Number of documents retrieved per query.
    batch_size : int
        Number of queries processed in a batch.
    threads : int
        Number of parallel threads for retrieval.
    tokenizer : SimpleTokenizer
        Tokenizer for answer matching.
    searcher : FaissSearcher
        FAISS searcher for document retrieval.

    Examples
    --------
    Basic Usage:

    >>> from rankify.dataset.dataset import Document, Question
    >>> from rankify.retrievers.retriever import Retriever
    >>> retriever = Retriever(method='dpr', model="dpr-multi", index_type="wiki", n_docs=5)
    >>> documents = [Document(question=Question("What is artificial intelligence?"))]
    >>> retrieved_documents = retriever.retrieve(documents)
    >>> print(retrieved_documents[0].contexts[0].text)
    """

    MSMARCO_CORPUS_URL = "https://huggingface.co/datasets/abdoelsayed/reranking-datasets/resolve/main/msmarco-passage-corpus/msmarco-passage-corpus.tsv?download=true"

    CACHE_DIR = os.environ.get("RERANKING_CACHE_DIR", "./cache" )
    DENSE_INDEX_MAP = {
        "dpr-multi": {
            "wiki": "wikipedia-dpr-100w.dpr-multi",
            "msmarco": "https://huggingface.co/datasets/abdoelsayed/reranking-datasets/resolve/main/index/msmarco_passage_dpr_multi.zip?download=true"
        },
        "dpr-single": {
            "wiki": "wikipedia-dpr-100w.dpr-single-nq",
            "msmarco": "https://huggingface.co/datasets/abdoelsayed/reranking-datasets/resolve/main/index/msmarco_passage_dpr_single.zip?download=true"
        },
        "ance-multi": {
            "wiki": "wikipedia-dpr-100w.ance-multi",
            "msmarco": "https://huggingface.co/datasets/abdoelsayed/reranking-datasets/resolve/main/index/msmarco_passage_ance.zip?download=true"
        },
        "bpr-single": {
            "wiki": "wikipedia-dpr-100w.bpr-single-nq",
            "msmarco": "https://huggingface.co/datasets/abdoelsayed/reranking-datasets/resolve/main/index/msmarco_passage_bpr.zip?download=true"
        }
    }
    QUERY_ENCODER_MAP = {
        "dpr-multi": "facebook/dpr-question_encoder-multiset-base",
        "dpr-single": "facebook/dpr-question_encoder-single-nq-base",
        "ance-multi": "castorini/ance-dpr-question-multi",
        "bpr-single": "castorini/bpr-nq-question-encoder"
    }

    def __init__(self, model: str = "dpr-multi", index_type: str = "wiki", n_docs: int = 10, batch_size: int = 36, threads: int = 30):
        """
        Initializes the DPR retriever with batch processing for efficiency.

        Parameters
        ----------
        model : str
            Type of retriever (e.g., "dpr-multi").
        index_type : str
            Dataset type ("wiki" or "msmarco").
        n_docs : int
            Number of documents to retrieve per query.
        batch_size : int
            Batch size for processing queries.
        threads : int
            Number of threads for batch search.
        """
        if model not in self.DENSE_INDEX_MAP:
            raise ValueError(f"Unsupported retriever model: {model}")

        if index_type not in self.DENSE_INDEX_MAP[model]:
            raise ValueError(f"Unsupported index type '{index_type}' for model '{model}'.")

        self.model = model
        self.index_type = index_type
        self.n_docs = n_docs
        self.batch_size = batch_size
        self.threads = threads
        self.tokenizer = SimpleTokenizer()

        # Load the appropriate searcher
        self.searcher = self._initialize_searcher()
        if index_type == "msmarco":
            self._load_msmarco_corpus()
    def _initialize_searcher(self):
        """
        Initializes the FaissSearcher for the given model and index type.

        Returns
        -------
        FaissSearcher
            Initialized FaissSearcher.
        """
        index_url_or_prebuilt = self.DENSE_INDEX_MAP[self.model][self.index_type]

        if index_url_or_prebuilt.startswith("http"):  # Handle downloadable indexes
            index_folder_name = os.path.basename(index_url_or_prebuilt).split("?")[0].replace(".zip", "")
            local_dir = os.path.join(self.CACHE_DIR, 'index' , index_folder_name ) 

            # Ensure the index is downloaded and extracted only once
            if not os.path.exists(local_dir):
                print(f"Preparing index folder at {local_dir}...")
                self._download_and_extract(index_url_or_prebuilt, local_dir)

            # Use the local directory to initialize FaissSearcher
            print(f"Initializing FaissSearcher with local index at {local_dir}...")
            return FaissSearcher(local_dir, self.QUERY_ENCODER_MAP[self.model])

        else:  # Handle prebuilt indexes from Pyserini
            print(f"Initializing FaissSearcher with prebuilt index: {index_url_or_prebuilt}...")
            return FaissSearcher.from_prebuilt_index(index_url_or_prebuilt, self.QUERY_ENCODER_MAP[self.model])

    def _load_msmarco_corpus(self):
        """
        Downloads and loads the MSMARCO passage corpus into memory.
        """
        corpus_file = os.path.join(self.CACHE_DIR , "msmarco-passage-corpus.tsv")
        if not os.path.exists(corpus_file):
            self._download_file(self.MSMARCO_CORPUS_URL, corpus_file)

        print("Loading MSMARCO corpus...")
        self.corpus = {}
        with open(corpus_file, "r", encoding="utf-8") as f:
            next(f)  # Skip header
            for line in f:
                doc_id, text, title = line.strip().split("\t")
                self.corpus[doc_id] = {"text": text, "title": title}
    def _download_file(self, url, save_path):
        """
        Downloads a file from the specified URL to the given path.
        """
        print(f"Downloading {os.path.basename(save_path)}...")
        response = requests.get(url, stream=True)
        with open(save_path, "wb") as f:
            for chunk in tqdm(response.iter_content(chunk_size=1024), desc=f"Downloading {os.path.basename(save_path)}"):
                f.write(chunk)

    def _download_and_extract(self, url, destination):
        """
        Downloads and extracts a ZIP file from a given URL.

        Parameters
        ----------
        url : str
            URL of the ZIP file.
        destination : str
            Destination directory.
        """
        # Ensure destination directory exists (only once here)
        os.makedirs(destination, exist_ok=True)

        zip_name = os.path.basename(url).split("?")[0]
        zip_path = os.path.join(self.CACHE_DIR, "index", zip_name)  # Temporarily save in CACHE_DIR

        if not os.path.exists(zip_path):
            print(f"Downloading index from {url}...")
            response = requests.get(url, stream=True)
            with open(zip_path, "wb") as f:
                for chunk in tqdm(response.iter_content(chunk_size=1024), desc="Downloading Index"):
                    f.write(chunk)

        print(f"Extracting {zip_name} into {destination}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(destination)

        os.remove(zip_path)  # Remove the ZIP file to save space
        print("Extraction complete.")




    def retrieve(self, documents: List[Document]) -> List[Document]:
        queries = [doc.question.question for doc in documents]
        #batch_results = self.searcher.batch_search(queries, [str(i) for i in range(len(queries))], k=self.n_docs, threads=self.threads)
        batch_results = self._batch_search(queries, [str(i) for i in range(len(queries))])
        for i, document in enumerate(tqdm(documents, desc="Processing documents", unit="doc")): #enumerate(documents):
            contexts = []
            hits = batch_results.get(str(i), [])
            for hit in hits:
                try:
                    if self.index_type == "msmarco":
                        doc_id = hit.docid
                        doc_data = self.corpus.get(doc_id, {"text": "Not Found", "title": "Not Found"})
                        text = doc_data["text"]
                        title = doc_data["title"]
                    else:
                        # Handle 'wiki' index
                        lucene_doc = self.searcher.doc(hit.docid)
                        raw_content = json.loads(lucene_doc.raw())
                        content = raw_content.get("contents", "")
                        title = content.split('\n')[0] if '\n' in content else "No Title"
                        text = content.split('\n')[1] if '\n' in content else content

                    context = Context(
                        id=int(hit.docid),
                        title=title,
                        text=text,
                        score=hit.score,
                        has_answer=has_answers(text, document.answers.answers, self.tokenizer)
                    )
                    contexts.append(context)
                except Exception as e:
                    print(f"Error processing document ID {hit.docid}: {e}")

            document.contexts = contexts

        return documents


    def _batch_search(self, queries, qids):
        """
        Performs batch search using Pyserini's batch_search method.

        Parameters
        ----------
        queries : List[str]
            List of query strings.
        qids : List[str]
            List of corresponding query IDs.

        Returns
        -------
        dict
            A dictionary where the keys are query IDs and the values are the retrieved hits.
        """
        batch_results = {}
        batch_qids, batch_queries = [], []

        for idx, (qid, query) in enumerate(tqdm(zip(qids, queries), desc="Batch search", total=len(qids))):
            query = self._preprocess_query(query)
            batch_qids.append(qid)
            batch_queries.append(query)

            if (idx + 1) % self.batch_size == 0 or idx == len(qids) - 1:
                try:
                    results = self.searcher.batch_search(batch_queries, batch_qids, self.n_docs, self.threads)
                    batch_results.update(results)
                except Exception as e:
                    #asddddd
                    print(f"Batch search failed for queries: {batch_queries} with error {e}")
                batch_qids.clear()
                batch_queries.clear()

        return batch_results

    def _preprocess_query(self, query):
        """
        Preprocess the query to ensure it does not exceed the maximum token length.

        Parameters:
        query (str): Input query string.

        Returns:
        str: Preprocessed query that is truncated to 512 tokens if necessary.
        """
        # Tokenize the query
        tokenized_query = tokenizer(query, add_special_tokens=True)

        # Check token length
        if len(tokenized_query["input_ids"]) > 511:
            #print(f"Query exceeds maximum token length (512). Truncating...")
            # Truncate to the first 512 tokens
            truncated_query = tokenizer.decode(tokenized_query["input_ids"][:511], skip_special_tokens=True)
            return truncated_query
        return query
