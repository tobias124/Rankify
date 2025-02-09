import torch
from rankify.models.base import BaseRanking
from rankify.dataset.dataset import Document
from rankify.utils.models.rank_llm.data import Request
from rankify.utils.models.rank_llm.rerank.listwise import RankListwiseOSLLM
from typing import List
from rankify.utils.models.rank_llm.rerank import PromptMode
from tqdm import tqdm  # Import tqdm for progress tracking

class ZephyrReranker(BaseRanking):
    """
    Implements **ZephyrReranker** `[37]`_ , a **listwise ranking approach** designed for
    **zero-shot passage reranking** with strong **robustness and efficiency**.

    .. _[37]: https://arxiv.org/abs/2312.02724

    This method utilizes a **RankZephyr-based model** to score query-passage relevance 
    and **reorder retrieved documents** based on contextualized ranking predictions.

    References
    ----------
    .. [37] Pradeep, R. et al. (2023). RankZephyr: Effective and Robust Zero-Shot Listwise Reranking is a Breeze!
 

    Attributes
    ----------
    method : str
        The name of the reranking method.
    model_name : str
        The name of the pre-trained **Zephyr** model used for reranking.
    window_size : int
        The **window size** used for batch-wise ranking.
    _reranker : RankListwiseOSLLM
        The Zephyr-based reranker instance.

    Examples
    --------
    Basic Usage:

    >>> from rankify.dataset.dataset import Document, Question, Context
    >>> from rankify.models.reranking import Reranking
    >>>
    >>> # Define a query and contexts
    >>> question = Question("What are the benefits of meditation?")
    >>> contexts = [
    >>>     Context(text="Meditation reduces stress and improves focus.", id=0),
    >>>     Context(text="Excessive noise pollution affects mental health.", id=1),
    >>>     Context(text="Daily meditation can enhance emotional well-being.", id=2),
    >>> ]
    >>> document = Document(question=question, contexts=contexts)
    >>>
    >>> # Initialize Zephyr Reranker
    >>> model = Reranking(method='zephyr_reranker', model_name='rank_zephyr_7b_v1_full')
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
        model_name: str = "castorini/rank_zephyr_7b_v1_full",
        **kwargs,
    ):
        """
        Initializes the ZephyrReranker for integration into the framework.

        Parameters
        ----------
        method : str
            Method name (optional).
        model_name : str
            Path to the Zephyr model.
        **kwargs : dict
            Additional keyword arguments for model configuration.
        """
        context_size: int = kwargs.get("context_size", 4096)
        prompt_mode = kwargs.get("prompt_mode", PromptMode.RANK_GPT)
        num_few_shot_examples: int = kwargs.get("num_few_shot_examples", 0)
        device: str = kwargs.get("device", "cuda")
        num_gpus: int = kwargs.get("num_gpus", 1)
        variable_passages: bool = kwargs.get("variable_passages", True)
        window_size: int = kwargs.get("window_size", 20)
        system_message: str = kwargs.get(
            "system_message",
            "You are RankLLM, an intelligent assistant that can rank passages based on their relevancy to the query",
        )

        self.window_size = window_size
        self._reranker = RankListwiseOSLLM(
            model=model_name,
            context_size=context_size,
            prompt_mode=prompt_mode,
            num_few_shot_examples=num_few_shot_examples,
            device=device,
            num_gpus=num_gpus,
            variable_passages=variable_passages,
            window_size=window_size,
            system_message=system_message,
        )

    def rank(self, documents: List[Document]) -> List[Document]:
        """
        Reranks each document's contexts using the Zephyr model.

        Parameters
        ----------
        documents : List[Document]
            A list of Document instances to rerank.

        Returns
        -------
        List[Document]
            The reranked list of Document instances with updated `reorder_contexts`.
        """
        for document in tqdm(documents, desc="Reranking Documents"):
            document = self._rerank_document(document)
        return documents

    def _rerank_document(self, document: Document) -> Document:
        """
        Rerank a single document using the Zephyr model.

        Parameters
        ----------
        document : Document
            A Document instance to rerank.

        Returns
        -------
        Document
            The reranked Document instance with updated `reorder_contexts`.
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

        # Rerank using the Zephyr model
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
