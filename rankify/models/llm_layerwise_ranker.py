from typing import Union, List, Optional
from rankify.models.base import BaseRanking
from rankify.dataset.dataset import Document
from transformers import AutoModelForCausalLM, AutoTokenizer

from rankify.utils.helper import get_device,get_dtype
import torch
from tqdm import tqdm  # Import tqdm for progress tracking

class LLMLayerWiseRanker(BaseRanking):
    """
    Implements LLM Layer-Wise Reranking `[13]`_  `[14]`_, a zero-shot ranking approach using large language models (LLMs).

    .. _[13]: https://arxiv.org/abs/2312.15503

    .. _[14]: https://arxiv.org/abs/2402.03216


    This method performs layer-wise scoring of query-passage pairs, leveraging self-knowledge distillation to 
    enhance ranking performance across multiple granularities.

    References
    ----------
    .. [13]  Li et al. (2023): Making Large Language Models A Better Foundation For Dense Retrieval.
    .. [14]  Chen et al. (2024): BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation.


    Attributes
    ----------
    model_name : str
        The name of the pre-trained model used for reranking.
    max_sequence_length : int
        The maximum token length of the input sequence (default: `512`).
    device : torch.device
        The device (CPU/GPU) on which the model runs.
    dtype : torch.dtype
        The tensor data type for model inference.
    batch_size : int
        The batch size for processing query-passage pairs.
    tokenizer : transformers.AutoTokenizer
        Tokenizer for the model.
    model : transformers.AutoModelForCausalLM
        The transformer-based LLM model for reranking.
    prompt : str
        The default prompt template used for ranking.
    params : dict
        Model-specific configuration parameters.

    Examples
    --------
    Basic usage:

    >>> from rankify.dataset.dataset import Document, Question, Context
    >>> from rankify.models.reranking import Reranking
    >>>
    >>> # Define a query and contexts
    >>> question = Question("What are the effects of climate change?")
    >>> contexts = [
    >>>     Context(text="Rising temperatures are causing ice caps to melt.", id=0),
    >>>     Context(text="Many species face extinction due to habitat loss.", id=1),
    >>>     Context(text="Ocean acidification is increasing.", id=2),
    >>> ]
    >>> document = Document(question=question, contexts=contexts)
    >>>
    >>> # Initialize LLM Layer-Wise Ranker
    >>> model = Reranking(method='llm_layerwise_ranker', model_name='bge-multilingual-gemma2')
    >>> model.rank([document])
    >>>
    >>> # Print reordered contexts
    >>> print("Reordered Contexts:")
    >>> for context in document.reorder_contexts:
    >>>     print(context.text)
    """
    
    # Class-level constants
    PROMPTS = {
        "BAAI/bge-reranker-v2.5-gemma2-lightweight": "Given a query A and a passage B, determine whether the passage contains an answer to the query by providing a prediction of either 'Yes' or 'No'.",
        "default": "Given a query A and a passage B, determine whether the passage contains an answer to the query by providing a prediction of either 'Yes' or 'No'.",
    }

    DEFAULT_PARAMS = {
        "default": {},
        "BAAI/bge-multilingual-gemma2": {},
        "BAAI/bge-reranker-v2-gemma": {},
        "BAAI/bge-reranker-v2-minicpm-layerwise": {"cutoff_layers": [28]},
        "BAAI/bge-reranker-v2.5-gemma2-lightweight": {
            "cutoff_layers": [28],
            "compress_ratio": 2,
            "compress_layer": [24, 40],
        },
    }

    def __init__(
        self, method = None, model_name = None, api_key = None, **kwargs):

        """
        Initializes the LLM Layer-Wise Ranker.

        Parameters
        ----------
        method : str, optional
            The name of the ranking method.
        model_name : str, optional
            The name of the pre-trained model (default: `"BAAI/bge-reranker-v2.5-gemma2-lightweight"`).
        api_key : str, optional
            API key for authentication (if needed).
        kwargs : dict
            Additional configuration arguments (e.g., `max_sequence_length`, `batch_size`, `device`, `dtype`).
        """
        
        max_sequence_length= kwargs.get("max_sequence_length", 512)
        device = kwargs.get("device", "cuda")
        self.device = get_device(device)
        self.batch_size = kwargs.get("batch_size", 16)
        
    
        self.model_name = model_name
        
        self.dtype = get_dtype( kwargs.get("dtype", torch.float32), self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.max_sequence_length = max_sequence_length
        self.tokenizer.model_max_length = self.max_sequence_length
        self.tokenizer.padding_side = "right"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype=self.dtype
        ).to(self.device)
        self.model.eval()

        self.params = self.DEFAULT_PARAMS.get(model_name, self.DEFAULT_PARAMS["default"])
        self.prompt = self.PROMPTS.get(model_name, self.PROMPTS["default"])

    def _get_inputs(self, pairs, max_sequence_length: int):
        prompt = self.prompt
        sep = "\n"
        prompt_inputs = self.tokenizer(
            prompt, return_tensors=None, add_special_tokens=False
        )["input_ids"]
        sep_inputs = self.tokenizer(sep, return_tensors=None, add_special_tokens=False)[
            "input_ids"
        ]
        inputs = []
        for query, passage in pairs:
            query_inputs = self.tokenizer(
                f"A: {query}",
                return_tensors=None,
                add_special_tokens=False,
                max_length=max_sequence_length * 3 // 4,
                truncation=True,
            )
            passage_inputs = self.tokenizer(
                f"B: {passage}",
                return_tensors=None,
                add_special_tokens=False,
                max_length=max_sequence_length,
                truncation=True,
            )
            item = self.tokenizer.prepare_for_model(
                [self.tokenizer.bos_token_id] + query_inputs["input_ids"],
                sep_inputs + passage_inputs["input_ids"],
                truncation="only_second",
                max_length=max_sequence_length,
                padding=False,
                return_attention_mask=False,
                return_token_type_ids=False,
                add_special_tokens=False,
            )
            item["input_ids"] = item["input_ids"] + sep_inputs + prompt_inputs
            item["attention_mask"] = [1] * len(item["input_ids"])
            inputs.append(item)

        return self.tokenizer.pad(
            inputs,
            padding=True,
            max_length=max_sequence_length + len(sep_inputs) + len(prompt_inputs),
            pad_to_multiple_of=8,
            return_tensors="pt",
        )

    @torch.no_grad()
    def rank(self, documents: List[Document]) -> List[Document]:
        """
        Ranks the contexts within each document based on their relevance to the document's query.

        Parameters
        ----------
        documents : list of Document
            The documents whose contexts need to be ranked.

        Returns
        -------
        list of Document
            The documents with reordered contexts.
        """
        for doc in tqdm(documents, desc="Reranking Documents"):
            query = doc.question.question
            pairs = [(query, context.text) for context in doc.contexts]
            batched_pairs = [
                pairs[i : i + self.batch_size] for i in range(0, len(pairs), self.batch_size)
            ]
            scores = []

            for batch in batched_pairs:
                inputs = self._get_inputs(batch, max_sequence_length=self.max_sequence_length)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                outputs = self.model(**inputs, return_dict=True, **self.params)
                all_scores = [scores[:, -1].view(-1,).float() for scores in outputs[0]]
                batch_scores = all_scores[-1].cpu().numpy().tolist()
                scores.extend(batch_scores)

            # Update each context with its score and reorder based on scores
            for context, score in zip(doc.contexts, scores):
                context.score = score

            doc.reorder_contexts = sorted(doc.contexts, key=lambda x: x.score, reverse=True)

        return documents

    @torch.no_grad()
    def score(self, query: str, doc: str) -> float:
        """
        Scores a single query-document pair.

        Parameters
        ----------
        query : str
            The query string.
        doc : str
            The document to be scored.

        Returns
        -------
        float
            The relevance score.
        """
        inputs = self._get_inputs([(query, doc)], max_sequence_length=self.max_sequence_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs, return_dict=True, **self.params)
        score = outputs.logits[:, -1].view(-1).float().item()
        return score
