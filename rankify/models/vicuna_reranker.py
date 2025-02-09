import torch
from rankify.models.base import BaseRanking
from rankify.dataset.dataset import Document
from rankify.utils.models.rank_llm.data import Request
from rankify.utils.models.rank_llm.rerank.listwise import RankListwiseOSLLM
from typing import List
from tqdm import tqdm  # Import tqdm for progress tracking


class VicunaReranker(BaseRanking):
    """
    Implements **RankVicuna** `[36]`_, a **zero-shot listwise document reranking** method using **Vicuna**.

    .. _[36]: https://arxiv.org/abs/2309.15088

    RankVicuna is a **listwise** ranking method that leverages open-source **large language models** (LLMs) 
    for **document reranking** without requiring fine-tuning.

    References
    ----------
    .. [36] Pradeep et al. (2023). RankVicuna: Zero-Shot Listwise Document Reranking with Open-Source LLMs.

    Attributes
    ----------
    method : str
        The reranking method name.
    model_name : str
        The name or path of the **Vicuna** model used for reranking.
    window_size : int
        The **window size** for listwise reranking.
    _reranker : RankListwiseOSLLM
        The **RankVicuna** reranking model instance.

    Examples
    --------
    Basic Usage:

    >>> from rankify.dataset.dataset import Document, Question, Context
    >>> from rankify.models.reranking import Reranking
    >>>
    >>> # Define a query and contexts
    >>> question = Question("What are the health benefits of green tea?")
    >>> contexts = [
    >>>     Context(text="Green tea is rich in antioxidants that improve heart health.", id=0),
    >>>     Context(text="Drinking green tea may boost brain function and alertness.", id=1),
    >>>     Context(text="Excessive sugar intake increases the risk of diabetes.", id=2),
    >>> ]
    >>> document = Document(question=question, contexts=contexts)
    >>>
    >>> # Initialize Vicuna reranker
    >>> model = Reranking(method='vicuna_reranker', model_name='rank_vicuna_7b_v1')
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
        model_name: str = "castorini/rank_vicuna_7b_v1", **kwargs
    ):
        """
        Initializes **RankVicuna** for listwise document reranking.

        Parameters
        ----------
        method : str, optional
            The reranking method name.
        model_name : str, optional
            The name of the **RankVicuna** model (default: `'castorini/rank_vicuna_7b_v1'`).
        kwargs : dict
            Additional parameters:
            - `context_size`: int (default=4096) - Maximum context size for the model.
            - `num_few_shot_examples`: int (default=0) - Number of few-shot examples.
            - `device`: str (default="cuda") - Computation device (CPU/GPU).
            - `num_gpus`: int (default=1) - Number of GPUs to use.
            - `variable_passages`: bool (default=False) - Whether to handle variable passage lengths.
            - `window_size`: int (default=20) - Sliding window size for reranking.
            - `system_message`: str (optional) - System message for the model.
        """
        context_size: int =  kwargs.get("context_size", 4096)
        num_few_shot_examples: int = kwargs.get("num_few_shot_examples", 0) 
        device: str = kwargs.get("device", "cuda") 
        num_gpus: int = kwargs.get("num_gpus", 1) 
        variable_passages: bool = kwargs.get("variable_passages", False) 
        window_size: int = kwargs.get("window_size", 20)
        system_message: str = kwargs.get("system_message", None)
        
        self.window_size = window_size
        self._reranker = RankListwiseOSLLM(
            model=model_name,
            context_size=context_size,
            num_few_shot_examples=num_few_shot_examples,
            device=device,
            num_gpus=num_gpus,
            variable_passages=variable_passages,
            window_size=window_size,
            system_message=system_message,
        )

    def rank(self, documents: List[Document]) -> List[Document]:
        """
        Reranks a list of **Document** instances using **RankVicuna**.

        Parameters
        ----------
        documents : List[Document]
            A list of **Document** instances to rerank.

        Returns
        -------
        List[Document]
            The reranked list of **Documents** with updated `reorder_contexts`.
        """
        for document in tqdm(documents, desc="Reranking Documents"):
            document = self._rerank_document(document)
        return documents

    def _rerank_document(self, document: Document):
        """
        Rerank a single document using **RankVicuna**.

        Parameters
        ----------
        document : Document
            A **Document** instance to rerank.

        Returns
        -------
        Document
            The reranked **Document** instance with updated `reorder_contexts`.
        """
        # Prepare request data structure for reranking
        request = Request(
            query={"text": document.question.question},  # Extract `text` from `Question`
            candidates=[
                {"docid": ctx.id, "doc": {"text": ctx.text}, "score": ctx.score}
                for ctx in document.contexts
            ],
        )

        rank_start = 0
        rank_end = 100
        window_size = self.window_size
        step = 10
        shuffle_candidates = False
        logging = False

        # Rerank using the Vicuna model
        reranked_result = self._reranker.rerank_batch(
            requests=[request],
            rank_start=rank_start,
            rank_end=rank_end,
            window_size=window_size,
            step=step,
            shuffle_candidates=shuffle_candidates,
            logging=logging,
        )[0]

        # Create a mapping from docid to the original context
        docid_to_context = {ctx.id: ctx for ctx in document.contexts}

        # Reorder contexts based on reranked_result
        document.reorder_contexts = [
            docid_to_context[int(candidate["docid"])]
            for candidate in reranked_result.candidates
        ]

        return document
