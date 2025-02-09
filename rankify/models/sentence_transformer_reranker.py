import torch
from sentence_transformers import SentenceTransformer
from rankify.models.base import BaseRanking
from rankify.dataset.dataset import Document
from typing import List
from tqdm import tqdm  # Import tqdm for progress tracking


class SentenceTransformerReranker(BaseRanking):
    """
    Implements **SentenceTransformerReranker** `[21]`_ `[22]`_ `[23]`_ `[24]`_ `[25]`_ , a **dense retrieval reranking approach**
    that uses **Sentence Transformers** for encoding queries and passages.

    .. _[21]: https://arxiv.org/abs/2112.07899
    .. _[22]: https://arxiv.org/abs/2002.10957
    .. _[23]: https://arxiv.org/abs/2108.08877
    .. _[24]: https://arxiv.org/abs/2108.05540
    .. _[25]: https://arxiv.org/abs/1908.10084

    This method **leverages large dual encoders**, **distilled self-attention**, and **corpus-aware pre-training**
    to enhance retrieval quality. It supports **Sentence-BERT (SBERT) embeddings**, **MiniLM**, and **Sentence-T5** for ranking.

    References
    ----------
    .. [21] Ni et al. (2021). Large Dual Encoders are Generalizable Retrievers.
    .. [22] Wang et al. (2020). MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression.
    .. [23] Ni et al. (2021). Sentence-T5: Scalable Sentence Encoders from Pre-trained Text-to-Text Models.
    .. [24] Gao & Callan (2021). Unsupervised Corpus Aware Language Model Pre-training for Dense Passage Retrieval.
    .. [25] Reimers (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.

    Attributes
    ----------
    method : str
        The name of the reranking method.
    model_name : str
        The name or path of the **Sentence Transformer** model.
    device : str
        The device (CPU/GPU) used for inference.
    query_prefix : str
        Prefix to prepend to query texts.
    document_prefix : str
        Prefix to prepend to document texts.
    normalize_embeddings : bool
        Whether to **normalize embeddings** before computing similarity.
    model : SentenceTransformer
        The **Sentence Transformer** model used for reranking.

    Examples
    --------
    Basic Usage:

    >>> from rankify.dataset.dataset import Document, Question, Context
    >>> from rankify.models.reranking import Reranking
    >>>
    >>> # Define a query and contexts
    >>> question = Question("What are the benefits of machine learning?")
    >>> contexts = [
    >>>     Context(text="Machine learning improves decision-making and automation.", id=0),
    >>>     Context(text="Quantum computing explores new paradigms in computation.", id=1),
    >>>     Context(text="Deep learning allows neural networks to learn from large data.", id=2),
    >>> ]
    >>> document = Document(question=question, contexts=contexts)
    >>>
    >>> # Initialize SentenceTransformerReranker
    >>> model = Reranking(method='sentence_transformer_reranker', model_name='all-MiniLM-L6-v2')
    >>> model.rank([document])
    >>>
    >>> # Print reordered contexts
    >>> print("Reordered Contexts:")
    >>> for context in document.reorder_contexts:
    >>>     print(context.text)
    """
    def __init__(
        self,
        method: str = None,
        model_name: str = "all-MiniLM-L6-v2",
        **kwargs
    ):
        """
        Initializes **Sentence Transformer Reranker** for reranking tasks.

        Parameters
        ----------
        method : str
            The name of the reranking method.
        model_name : str
            The name or path to the **Sentence Transformer** model.
        kwargs : dict
            Additional keyword arguments for customization:
            - device: str (default="auto") - computation device.
            - query_prefix: str (default="") - prefix for query texts.
            - document_prefix: str (default="") - prefix for document texts.
            - use_fp16: bool (default=True) - whether to use **FP16** inference.
            - normalize_embeddings: bool (default=True) - whether to **normalize embeddings**.
            - max_seq_length: int (default=512) - maximum tokenization length.
        """
        super().__init__(method)
        self.device = self._detect_device(kwargs.get("device", "auto"))
        if model_name =="other":
            model_name =  kwargs.get("name", "all-MiniLM-L6-v2")
        self.model = SentenceTransformer(
            model_name, device=self.device, trust_remote_code=True
        )
        if kwargs.get("use_fp16", True) and "cuda" in self.device:
            self.model.half()

        self.query_prefix = kwargs.get("query_prefix", "")
        self.document_prefix = kwargs.get("document_prefix", "")
        self.normalize_embeddings = kwargs.get("normalize_embeddings", True)
        self.model.max_seq_length = kwargs.get("max_seq_length", 512)

    def rank(self, documents: List[Document]) -> List[Document]:
        """
        Reranks a list of **Document** instances based on **Sentence Transformer** similarity.

        Parameters
        ----------
        documents : List[Document]
            A list of `Document` instances to rerank.

        Returns
        -------
        List[Document]
            The documents with updated `reorder_contexts`.
        """
        for document in tqdm(documents, desc="Reranking Documents"):
            document = self._rerank_document(document)
        return documents

    def _rerank_document(self, document: Document) -> Document:
        """
        Reranks a single document's contexts using the **Sentence Transformer** model.

        Parameters
        ----------
        document : Document
            A **Document** instance to rerank.

        Returns
        -------
        Document
            The reranked **Document** with updated `reorder_contexts`.
        """
        query = document.question.question  # Extract query text
        contexts = document.contexts

        # Extract context texts
        context_texts = [ctx.text for ctx in contexts]

        # Compute relevance scores
        scores = self._rerank(query, context_texts)

        # Map scores back to contexts and update the scores
        for ctx, score in zip(contexts, scores):
            ctx.score = score

        # Map scores back to contexts
        scored_contexts = list(zip(contexts, scores))
        scored_contexts.sort(key=lambda x: x[1], reverse=True)

        # Reorder contexts in the document
        document.reorder_contexts = [ctx for ctx, _ in scored_contexts]
        return document

    def _rerank(self, query: str, documents: List[str]) -> List[float]:
        """
        Computes similarity scores between query and documents.

        Parameters
        ----------
        query : str
            The **query** text.
        documents : List[str]
            A list of **document texts**.

        Returns
        -------
        List[float]
            Similarity scores for each document.
        """
        # Add prefixes to query and documents
        query_text = self.query_prefix + query
        documents = [self.document_prefix + doc for doc in documents]

        # Encode query and documents
        embeddings = self.model.encode(
            [query_text] + documents,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_tensor=True,
            show_progress_bar=False  # Disable tqdm progress bar
        )
        query_emb = embeddings[0]
        document_embs = embeddings[1:]

        # Compute cosine similarity scores
        scores = torch.nn.functional.cosine_similarity(
            query_emb.unsqueeze(0), document_embs
        ).tolist()
        return scores

    @staticmethod
    def _detect_device(device: str) -> str:
        """
        Detects the appropriate device for computation.

        Parameters
        ----------
        device : str
            Desired device ("auto", "cuda", or "cpu").

        Returns
        -------
        str
            The detected device.
        """
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
