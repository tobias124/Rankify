import logging
import math
from typing import List, Tuple, Union, Optional
from transformers import T5ForConditionalGeneration, T5Tokenizer, GenerationConfig
from rankify.models.base import BaseRanking
from rankify.dataset.dataset import Document, Context
from rankify.utils.helper import get_device, get_dtype
import torch
from rankify.utils.pre_defind_models import PREDICTION_TOKENS
logger = logging.getLogger(__name__)
from tqdm import tqdm  # Import tqdm for progress tracking




class MonoT5(BaseRanking):
    """
    Implements **MonoT5 Reranking** `[17]`_, a **sequence-to-sequence (seq2seq) ranking approach** 
    using **T5** to assess document relevance in **zero-shot and fine-tuned settings**.

    .. _[17]: https://arxiv.org/abs/2003.06713

    This method **uses a pretrained MonoT5 model** to score query-passage relevance 
    and **reorder retrieved documents** based on learned representations.

    References
    ----------
    .. [17] Nogueira et al. (2020): Document Ranking with a Pretrained Sequence-to-Sequence Model.
 

    Attributes
    ----------
    method : str
        The name of the reranking method.
    model_name : str
        The name of the pre-trained **MonoT5** model used for reranking.
    device : torch.device
        The device (CPU/GPU) on which the model runs.
    _context_size : int
        The **maximum sequence length** for encoding.
    _batch_size : int
        The batch size used for efficient inference.
    _prompt_mode : str
        The type of **MonoT5 prompt template** being used.
    _tokenizer : T5Tokenizer
        The tokenizer for MonoT5.
    _llm : T5ForConditionalGeneration
        The pretrained **T5 model** for ranking.
    token_true_id : int
        Token ID corresponding to the **"true"** label.
    token_false_id : int
        Token ID corresponding to the **"false"** label.

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
    >>> # Initialize MonoT5 Reranker
    >>> model = Reranking(method='monot5', model_name='monot5-base-msmarco')
    >>> model.rank([document])
    >>>
    >>> # Print reordered contexts
    >>> print("Reordered Contexts:")
    >>> for context in document.reorder_contexts:
    >>>     print(context.text)
    """

    def __init__(self, method=None, model_name=None, **kwargs):
        """
        Initializes MonoT5 for reranking tasks.

        Parameters
        ----------
        method : str, optional
            The reranking method name.
        model_name : str, optional
            The name of the pretrained **MonoT5** model.
        kwargs : dict
            Additional parameters such as `batch_size` and `context_size`.
        """        
        device = kwargs.get("device", "cuda")
        self._device = get_device(device)
        self._context_size = kwargs.get("context_size", 512) 
        self._batch_size = kwargs.get("batch_size", 16)
        self._prompt_mode = kwargs.get("device", "monot5")
        self.inputs_template = kwargs.get("inputs_template", "Query: {query} Document: {text} Relevant:")

        # Load the T5 model and tokenizer
        self._tokenizer = T5Tokenizer.from_pretrained(model_name)
        self._llm = T5ForConditionalGeneration.from_pretrained(model_name).to(self._device)

        # Set tokens for scoring based on the model
        token_false, token_true = self._get_output_tokens(model_name)
        self.token_false_id = self._tokenizer.convert_tokens_to_ids(token_false)
        self.token_true_id = self._tokenizer.convert_tokens_to_ids(token_true)
        logger.info(f"True token ID set to {self.token_true_id}, False token ID set to {self.token_false_id}")

    def _get_scores(self, query: str, docs: List[str], max_length: int = 512) -> List[float]:
        """
        Calculate relevance scores for a list of documents given a query.

        Parameters
        ----------
        query : str
            The query text.
        docs : List[str]
            A list of document texts.
        max_length : int, optional
            The maximum sequence length for the model (default: 512).

        Returns
        -------
        List[float]
            Relevance scores for each document.
        """
        scores = []
        for batch in self._chunks(docs, self._batch_size):
            prompts = [self.inputs_template.format(query=query, text=text) for text in batch]
            tokenized = self._tokenizer(prompts, padding=True, truncation=True, return_tensors="pt", max_length=max_length).to(self._device)
            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]

            _, batch_scores = self._greedy_decode(
                model=self._llm,
                input_ids=input_ids,
                length=1,
                attention_mask=attention_mask,
                return_last_logits=True,
            )
            batch_scores = batch_scores[:, [self.token_false_id, self.token_true_id]].cpu()
            batch_scores = torch.log_softmax(batch_scores, dim=-1)
            scores.extend(torch.exp(batch_scores[:, 1]).tolist())
        return scores

    @torch.no_grad()
    def _greedy_decode(self, model, input_ids, length, attention_mask=None, return_last_logits=True):
        """
        Greedy decode implementation to get logits for true/false predictions.

        Parameters
        ----------
        model : T5ForConditionalGeneration
            The T5 model.
        input_ids : torch.Tensor
            Input tensor containing tokenized query-document pairs.
        length : int
            Number of decoding steps (typically 1).
        attention_mask : torch.Tensor, optional
            Attention mask to handle padding (default: None).
        return_last_logits : bool, optional
            Whether to return the final logits (default: True).

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            - **decode_ids**: Decoded token predictions.
            - **next_token_logits**: Logits of the last token.
        """
        decode_ids = torch.full((input_ids.size(0), 1), model.config.decoder_start_token_id, dtype=torch.long).to(input_ids.device)
        encoder_outputs = model.get_encoder()(input_ids, attention_mask=attention_mask)
        next_token_logits = None
        for _ in range(length):
            model_inputs = model.prepare_inputs_for_generation(decode_ids, encoder_outputs=encoder_outputs, attention_mask=attention_mask, use_cache=True)
            outputs = model(**model_inputs)
            next_token_logits = outputs[0][:, -1, :]  # Get logits for last token
            decode_ids = torch.cat([decode_ids, next_token_logits.max(1)[1].unsqueeze(-1)], dim=-1)
        return decode_ids, next_token_logits

    def rank(self, documents: List[Document]) -> List[Document]:
        """
        Ranks contexts within each document based on relevance to the query.

        Parameters
        ----------
        documents : List[Document]
            A list of `Document` instances to rerank.

        Returns
        -------
        List[Document]
            Documents with updated `reorder_contexts` after reranking.
        """
        for doc in tqdm(documents, desc="Reranking Documents"):
            query = doc.question.question
            prompts = [self.inputs_template.format(query=query, text=context.text) for context in doc.contexts]
            scores = self._get_scores(query, [context.text for context in doc.contexts])
            for context, score in zip(doc.contexts, scores):
                context.score = score
            doc.reorder_contexts = sorted(doc.contexts, key=lambda x: x.score, reverse=True)
        return documents
    @staticmethod
    def _chunks(contexts: list[Context],batch_size: int):
        """
        Splits a list of contexts into smaller batches.

        Parameters
        ----------
        contexts : List[Context]
            The list of contexts to split.
        batch_size : int
            The batch size.

        Yields
        ------
        List[Context]
            A batch of contexts.
        """
        for i in range(0, len(contexts), batch_size):
            yield contexts[i:i+batch_size]

    @staticmethod
    def _get_output_tokens(model_name: str, token_false: str = "auto", token_true: str = "auto"):
        if token_false == "auto" and model_name in PREDICTION_TOKENS:
            token_false = PREDICTION_TOKENS[model_name][0]
        if token_true == "auto" and model_name in PREDICTION_TOKENS:
            token_true = PREDICTION_TOKENS[model_name][1]
        if token_false == "auto" or token_true == "auto":
            token_false, token_true = PREDICTION_TOKENS["default"]
            logger.warning(f"Model {model_name} not found in PREDICTION_TOKENS. Using default tokens.")
        return token_false, token_true