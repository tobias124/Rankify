import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from rankify.models.base import BaseRanking
from rankify.dataset.dataset import Document
from typing import List
from rankify.utils.models.twolar_utils import Score
import copy
from tqdm import tqdm  # Import tqdm for progress tracking


class TWOLAR(BaseRanking):
    """
    Implements **TWOLAR** `[35]`_ , a **two-step LLM-augmented distillation method** for passage reranking.

    .. _[35]: https://arxiv.org/abs/2403.17759

    TWOLAR improves passage ranking **via a distillation method** that first generates an LLM-augmented score 
    and then refines it using a learned ranking model. 

    References
    ----------
    .. [35] Baldelli et al. (2024). TWOLAR: A TWO-step LLM-Augmented Distillation Method for Passage Reranking.

    Attributes
    ----------
    method : str
        The reranking method name.
    model_name : str
        The name or path of the **TWOLAR** pretrained model.
    device : torch.device
        The computation device (**CPU/GPU**).
    tokenizer : AutoTokenizer
        The tokenizer for encoding queries and passages.
    model : AutoModelForSeq2SeqLM
        The **TWOLAR** reranking model.
    batch_size : int
        The batch size for inference.
    max_length : int
        The **maximum sequence length** for encoding passages.
    score_strategy : str
        The strategy used to compute ranking scores.

    Examples
    --------
    Basic Usage:

    >>> from rankify.dataset.dataset import Document, Question, Context
    >>> from rankify.models.reranking import Reranking
    >>>
    >>> # Define a query and contexts
    >>> question = Question("What are the effects of climate change?")
    >>> contexts = [
    >>>     Context(text="Climate change leads to rising sea levels and extreme weather.", id=0),
    >>>     Context(text="Renewable energy helps reduce carbon emissions.", id=1),
    >>>     Context(text="Deforestation accelerates global warming.", id=2),
    >>> ]
    >>> document = Document(question=question, contexts=contexts)
    >>>
    >>> # Initialize TWOLAR reranker
    >>> model = Reranking(method='twolar', model_name='twolar-xl')
    >>> model.rank([document])
    >>>
    >>> # Print reordered contexts
    >>> print("Reordered Contexts:")
    >>> for context in document.reorder_contexts:
    >>>     print(context.text)
    """

    def __init__(self, method: str = None, model_name: str = None, api_key: str = None, **kwargs):
        """
        Initializes **TWOLAR** for reranking tasks.

        Parameters
        ----------
        method : str, optional
            The reranking method name.
        model_name : str, optional
            The name of the pretrained **TWOLAR** model.
        api_key : str, optional
            API key if required (default: None).
        kwargs : dict
            Additional parameters such as `batch_size`, `max_length`, and `score_strategy`.
        """
        self.method = method
        self.model_name = model_name or "Dundalia/TWOLAR-xl"
        self.api_key = api_key
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = kwargs.get("batch_size", 8)
        self.max_length = kwargs.get("max_length", 500)
        self.score_strategy = kwargs.get("score_strategy", "difference")

        # Load model and tokenizer
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.get_score = getattr(Score, self.score_strategy)

    def rank(self, documents: List[Document]) -> List[Document]:
        """
        Reranks a list of **Document** instances using **TWOLAR**.

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
            query = document.question.question
            contexts = [ctx.text for ctx in document.contexts]

            # Encode the inputs
            input_features = self._prepare_inputs(query, contexts)

            # Perform inference
            logits = self._inference(input_features)

            # Compute scores
            scores = self.get_score(logits).tolist()

            # Assign scores to contexts and sort
            copy_context = copy.deepcopy(document.contexts)
            for i, context in enumerate(copy_context):
                context.score = scores[i]
            
            ranked_contexts = sorted(copy_context, key=lambda ctx: ctx.score, reverse=True)

            document.reorder_contexts = ranked_contexts
            #reranked_documents.append(document)

        return documents

    def _prepare_inputs(self, query: str, contexts: List[str]):
        """
        Prepares the inputs for the model in **TWOLAR** format.

        Parameters
        ----------
        query : str
            The query text.
        contexts : List[str]
            A list of passage texts.

        Returns
        -------
        dict
            Tokenized input features for the model.
        """
        inputs = [f"Query: {query} Document: {ctx} Relevant: " for ctx in contexts]
        features = self.tokenizer(
            inputs,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length,
            padding=True,
        )
        features["input_ids"] = features.input_ids.to(self.device)
        features["attention_mask"] = features.attention_mask.to(self.device)
        features["decoder_input_ids"] = torch.full(
            (features.input_ids.size(0), 1),
            self.model.config.decoder_start_token_id,
            dtype=torch.long,
            device=self.device,
        )
        return features

    def _inference(self, features):
        """
        Performs inference on the given input features and returns logits.

        Parameters
        ----------
        features : dict
            Tokenized input features.

        Returns
        -------
        torch.Tensor
            Logits from the model.
        """
        with torch.no_grad():
            output = self.model(
                input_ids=features["input_ids"],
                attention_mask=features["attention_mask"],
                decoder_input_ids=features["decoder_input_ids"],
            )
        return output.logits[:, 0, :]

