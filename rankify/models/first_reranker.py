import torch
from rankify.models.base import BaseRanking
from rankify.dataset.dataset import Document
from rankify.utils.models.rank_listwise_os_llm import RankListwiseOSLLM,PromptMode,Result,FirstReranker
from typing import List
from enum import Enum
from tqdm import tqdm  # Import tqdm for progress tracking

    
class FirstModelReranker(BaseRanking):
    """
    Implements FIRST: Faster Improved Listwise Reranking with Single Token Decoding `[6]`_.

    .. _[6]: https://arxiv.org/abs/2406.15657

    FIRST is a listwise reranking model that uses an optimized window-based decoding approach
    to efficiently rank passages with single-token predictions rather than full-text decoding.

    This method allows scalable and efficient reranking, significantly improving speed and accuracy
    while maintaining high retrieval effectiveness.

    References
    ----------
    .. [6] Gangi Reddy, R., Doo, J., Xu, Y., Sultan, M. A., Swain, D., Sil, A., & Ji, H. (2024). FIRST: Faster Improved Listwise Reranking with Single Token Decoding. Proceedings of EMNLP 2024, pages 8642–8652.

    Attributes
    ----------
    method : str, optional
        The reranking method name.
    model_name : str
        The name of the model used for reranking.
    api_key : str, optional
        API key for accessing remote models (if applicable).
    context_size : int
        Maximum input length for the reranking model (default: `4096` tokens).
    top_k : int
        Number of top-ranked passages to retain after reranking (default: `20`).
    window_size : int
        Window size for the **sliding window decoding approach** (default: `9`).
    step_size : int
        Step size for moving the window during listwise ranking (default: `9`).
    use_logits : bool
        Whether to use **logits-based scoring** instead of rank ordering (default: `False`).
    use_alpha : bool
        Whether to apply **adaptive alpha scaling** in ranking (default: `False`).
    batched : bool
        Whether to use **batched ranking** for efficiency (default: `False`).
    device : str
        The computing device (`"cuda"` if available, otherwise `"cpu"`).
    agent : RankListwiseOSLLM
        The ranking model instance used for passage ranking.

    See Also
    --------
    Reranking : Main interface for reranking models, including `FirstModelReranker`.

    Examples
    --------
    Basic usage with the `Reranking` class:

    >>> from rankify.dataset.dataset import Document, Question, Answer, Context
    >>> from rankify.models.reranking import Reranking
    >>>
    >>> question = Question("Who invented the first light bulb?")
    >>> answers = Answer(["Thomas Edison is credited with inventing the first practical light bulb."])
    >>> contexts = [
    >>>     Context(text="Nikola Tesla contributed to AC electricity.", id=0),
    >>>     Context(text="Thomas Edison patented the first practical light bulb.", id=1),
    >>>     Context(text="Light bulbs use tungsten filaments.", id=2),
    >>>     Context(text="The Wright brothers invented the airplane.", id=3),
    >>> ]
    >>> document = Document(question=question, answers=answers, contexts=contexts)
    >>>
    >>> # Initialize Reranking with FirstModelReranker
    >>> model = Reranking(method='first_ranker', model_name='base')
    >>> model.rank([document])
    >>>
    >>> # Print reordered contexts
    >>> print("Reordered Contexts:")
    >>> for context in document.reorder_contexts:
    >>>     print(context.text)

    Notes
    -----
    - FIRST uses a windowed approach to process large query-document pairs efficiently.
    - Integrates into the `Reranking` class, so use `Reranking` instead of `FirstModelReranker` directly.
    - Uses single-token decoding for fast and effective ranking.
    """
    def __init__(self, method: str = None, model_name: str = None, api_key: str = None, **kwargs):
        """
        Initializes the FIRST model reranker.

        Parameters
        ----------
        method : str, optional
            The reranking method name.
        model_name : str
            The name of the reranking model to be used.
        api_key : str, optional
            API key for remote access (if applicable).
        kwargs : dict
            Additional parameters for model configuration.
        """
        self.method = method
        self.model_name = model_name
        self.api_key = api_key
        self.context_size = kwargs.get("context_size", 4096)
        self.top_k = kwargs.get("top_k", 20)
        self.window_size = kwargs.get("window_size", 4)
        self.step_size = kwargs.get("step_size", 2)
        self.use_logits = kwargs.get("use_logits", False)
        self.use_alpha = kwargs.get("use_alpha", False)
        self.batched = kwargs.get("batched", False)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.agent = self._initialize_agent()

    def _initialize_agent(self):
        """
        Initializes the RankListwiseOSLLM agent for reranking with the specified model and parameters.

        Returns
        -------
        RankListwiseOSLLM
            A listwise reranking model instance.
        """
        return RankListwiseOSLLM(
            model=self.model_name,
            context_size=self.context_size,
            prompt_mode=PromptMode.RANK_GPT,
            num_few_shot_examples=0,
            device=self.device,
            num_gpus=1,
            variable_passages=True,
            window_size=self.window_size,
            system_message="You are RankLLM, an intelligent assistant that can rank passages based on their relevancy to the query",
            batched=self.batched,
            max_model_len=8192
        )

    def rank(self, documents: List[Document]) -> List[Document]:
        """
        Reranks a list of documents using the FIRST reranking model.

        Parameters
        ----------
        documents : list of Document
            A list of Document instances to rerank.

        Returns
        -------
        list of Document
            Documents with updated `reorder_contexts` after reranking.

        Raises
        ------
        ValueError
            If no contexts are provided for reranking.
        """
        for document in tqdm(documents, desc="Reranking Documents"):
            self._rerank_document(document)
        return documents

    def _rerank_document(self, document: Document):
        """
        Applies the FIRST reranking model to reorder the document contexts.

        Parameters
        ----------
        document : Document
            A Document instance to be reranked.

        Returns
        -------
        None
            Updates `document.reorder_contexts` in place.
        """
        result = Result(
            query=document.question.question,
            hits=[{"docid": ctx.id, "content": ctx.text , "rank":  0, 'score':0} for ctx in document.contexts]
        )
        #print(result)
        # Perform reranking using `FirstReranker`
        reranker = FirstReranker(agent=self.agent)
        reranked_result = reranker.rerank(
            retrieved_result=result,
            use_logits=self.use_logits,
            use_alpha=self.use_alpha,
            rank_start=0,
            rank_end=self.top_k,
            window_size=self.window_size,
            step=self.step_size,
            logging=False,
            batched=self.batched
        )
        #print(reranked_result)
        # Update `Document.reorder_contexts` based on reranked `hits`
        context_dict = {ctx.id: ctx for ctx in document.contexts}

        document.reorder_contexts = [
            context_dict[hit["docid"]] for hit in reranked_result.hits if hit["docid"] in context_dict
        ]

