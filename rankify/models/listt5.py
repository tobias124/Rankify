import torch
from transformers import T5Tokenizer
from rankify.models.base import BaseRanking
from rankify.dataset.dataset import Document
from typing import List
import copy

from rankify.utils.models.fidt5 import FiDT5
from tqdm import tqdm  # Import tqdm for progress tracking



class ListT5(BaseRanking):
    """
    Implements ListT5 `[11]`_, a listwise reranker based on Fusion-in-Decoder (FiD) models.

    .. _[11]: https://arxiv.org/abs/2402.15838

    ListT5 ranks passages in chunks using a tournament-style sorting approach. The model processes 
    query-context pairs in batches, extracts ranking scores, and produces reordered results.

    References
    ----------
    .. [11] Yoon et al. (2024): ListT5: Listwise reranking with fusion-in-decoder improves zero-shot retrieval.

    Attributes
    ----------
    method : str, optional
        The reranking method name.
    model_name : str
        Name of the pre-trained ListT5 model.
    api_key : str, optional
        API key for authentication (if needed).
    use_gpu : bool
        Whether to use GPU acceleration.
    batch_size : int
        Number of query-document pairs processed in a batch.
    max_length : int
        Maximum tokenized sequence length.
    padding : str
        Padding strategy for tokenization (`"max_length"` or `"longest"`).
    listwise_k : int
        Number of contexts per chunk during listwise ranking.
    out_k : int
        Number of top contexts retained from each chunk.
    model : FiDT5
        The pre-trained Fusion-in-Decoder model for ranking.
    tokenizer : T5Tokenizer
        The tokenizer associated with the ListT5 model.

    See Also
    --------
    Fusion-in-Decoder : The FiD model architecture for improved document fusion.

    Examples
    --------
    Basic usage:

    >>> from rankify.dataset.dataset import Document, Question, Context
    >>> from rankify.models.reranking import Reranking
    >>>
    >>> question = Question("What are the effects of climate change?")
    >>> contexts = [
    >>>     Context(text="Rising temperatures are causing ice caps to melt.", id=0),
    >>>     Context(text="Many species face extinction due to habitat loss.", id=1),
    >>>     Context(text="Ocean acidification is increasing.", id=2),
    >>> ]
    >>> document = Document(question=question, contexts=contexts)
    >>>
    >>> # Initialize Reranking with ListT5
    >>> model = Reranking(method='listt5', model_name='listt5-base')
    >>> model.rank([document])
    >>>
    >>> # Print reordered contexts
    >>> print("Reordered Contexts:")
    >>> for context in document.reorder_contexts:
    >>>     print(context.text)

    Notes
    -----
    - Uses listwise ranking instead of pointwise or pairwise methods.
    - Processes passages in chunks to maintain model efficiency.
    - Supports batch processing for large-scale reranking tasks.
    """
    def __init__(self, method: str = None, model_name: str = None, api_key: str = None,  **kwargs):
        """
        Initializes ListT5 for **listwise document reranking**.

        Parameters
        ----------
        method : str, optional
            The reranking method name.
        model_name : str
            Name of the pre-trained ListT5 model.
        api_key : str, optional
            API key for authentication (if needed).
        kwargs : dict
            Additional parameters, including:
            - `batch_size`: Number of query-document pairs per batch (default: `20`).
            - `max_length`: Maximum sequence length for tokenization (default: `512`).
            - `padding`: Padding strategy (`"max_length"` or `"longest"`, default: `"max_length"`).
            - `listwise_k`: Number of contexts per chunk (default: `5`).
            - `out_k`: Number of top contexts retained per chunk (default: `2`).
        """
        self.method = method
        self.model_name = model_name
        self.api_key = api_key
        self.use_gpu = torch.cuda.is_available()
        self.batch_size = kwargs.get("batch_size", 20)
        self.max_length = kwargs.get("max_length", 512)
        self.padding = kwargs.get("padding", "max_length")
        self.listwise_k = kwargs.get("listwise_k", 5)  # Max contexts per chunk
        self.out_k = kwargs.get("out_k", 2)  # Top contexts per chunk
        self.model, self.tokenizer = self._load_model()

    def _load_model(self):
        """
        Loads the **pre-trained ListT5 model** and tokenizer.

        Returns
        -------
        tuple
            A tuple containing:
            - `model`: FiDT5 model.
            - `tokenizer`: T5Tokenizer instance.
        """
        tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        model = FiDT5.from_pretrained(self.model_name)
        if self.use_gpu:
            model = model.cuda()
        model.eval()
        return model, tokenizer

    def rank(self, documents: list[Document]):
        """
        Reranks documents using **ListT5’s listwise tournament sorting approach**.

        Each document’s contexts are processed in **chunks**, ranked, and merged.

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
            reordered_contexts = self._run_tournament_sort(document)
            document.reorder_contexts = reordered_contexts
        return documents

    def _run_tournament_sort(self, document):
        """
        Applies **tournament-style ranking** on a document's contexts.

        Parameters
        ----------
        document : Document
            The document whose contexts will be reranked.

        Returns
        -------
        List[Context]
            The reordered list of contexts.
        """
        question = document.question.question
        contexts = [ctx.text for ctx in document.contexts]
        full_list_idx = list(range(len(contexts)))
        saved_top_indices = []
        
        # Chunk processing based on listwise_k
        chunks = [full_list_idx[i:i + self.listwise_k] for i in range(0, len(full_list_idx), self.listwise_k)]
        for chunk in chunks:
            if len(chunk) < self.listwise_k:
                chunk = self._pad_chunk(chunk, full_list_idx)  # Ensure chunk has listwise_k items
            top_indices = self._get_out_k(question, contexts, chunk)
            saved_top_indices.extend(top_indices[:self.out_k])

        # Deduplicate and fill remaining slots
        saved_top_indices = self._remove_duplicates(saved_top_indices)
        if len(saved_top_indices) < len(full_list_idx):
            remaining_indices = [idx for idx in full_list_idx if idx not in saved_top_indices]
            saved_top_indices.extend(remaining_indices)

        return [document.contexts[idx] for idx in saved_top_indices]

    def _get_out_k(self, question, contexts, chunk):
        """
        Extracts **top-k relevant passages** from a chunk.

        Parameters
        ----------
        question : str
            The query text.
        contexts : List[str]
            List of passage texts.
        chunk : List[int]
            Indices of passages in the chunk.

        Returns
        -------
        List[int]
            Indices of top-ranked passages.
        """
        chunk_texts = self._prepare_texts(question, [contexts[i] for i in chunk])
        input_tensors = self._prepare_inputs(chunk_texts)
        output = self._run_inference(input_tensors)
        ranked_indices = self._get_rel_index(output, k=self.listwise_k)
        
        # Map relative indices to original chunk indices
        return [chunk[i - 1] for i in ranked_indices]

    def _prepare_texts(self, question, contexts):
        return [f"Query: {question}, Index: {i + 1}, Context: {ctx}" for i, ctx in enumerate(contexts)]

    def _prepare_inputs(self, texts):
        encoding = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=self.padding,
            max_length=self.max_length,
            truncation=True
        )
        input_tensors = {
            "input_ids": encoding["input_ids"].unsqueeze(0),
            "attention_mask": encoding["attention_mask"].unsqueeze(0)
        }
        if self.use_gpu:
            input_tensors = {key: tensor.cuda() for key, tensor in input_tensors.items()}
        return input_tensors

    def _run_inference(self, input_tensors):
        """
        Runs inference using **ListT5's decoder**.

        Parameters
        ----------
        input_tensors : dict
            Tokenized inputs for the model.

        Returns
        -------
        torch.Tensor
            Output tensor containing rankings.
        """
        output = self.model.generate(
            **input_tensors,
            max_length=self.listwise_k + 2,
            return_dict_in_generate=True,
            output_scores=True
        )
        return output

    def _get_rel_index(self, output, k):
        generated_text = self.tokenizer.batch_decode(output.sequences, skip_special_tokens=True)[0]
        return [int(idx) for idx in generated_text.split()[-k:]]

    def _pad_chunk(self, chunk, full_list):
        padded_chunk = chunk[:]
        i = 0
        while len(padded_chunk) < self.listwise_k:
            padded_chunk.append(full_list[i % len(full_list)])
            i += 1
        return padded_chunk

    def _remove_duplicates(self, indices):
        """
        Removes duplicate indices while maintaining order.

        Parameters
        ----------
        indices : List[int]
            List of ranked indices.

        Returns
        -------
        List[int]
            Deduplicated list of indices.
        """
        seen = set()
        unique_indices = []
        for idx in indices:
            if idx not in seen:
                seen.add(idx)
                unique_indices.append(idx)
        return unique_indices





class ListT5_old(BaseRanking):
    """
    A class to implement the ListT5 model for reranking using the Fusion-in-Decoder (FiD) approach.

    Attributes
    ----------
    method : str
        The name of the reranking method.
    model_name : str
        The name of the model being used.
    model : transformers.PreTrainedModel
        The pre-trained model used for reranking.
    tokenizer : transformers.PreTrainedTokenizer
        The tokenizer associated with the model.
    use_gpu : bool
        Whether to use GPU for computations.
    batch_size : int
        Batch size for processing contexts.
    """
    def __init__(self, method: str = None, model_name: str = None, api_key: str = None, **kwargs):
        """
        Initializes a ListT5 instance.

        Parameters
        ----------
        method : str, optional
            The name of the reranking method (default is None).
        model_name : str, optional
            The name of the model to be used for reranking (default is None).
        api_key : str, optional
            API key for accessing remote models or services (default is None).
        kwargs : dict
            Additional keyword arguments for model and tokenizer configurations.
        """
        self.method = method
        self.model_name = model_name
        self.api_key = api_key
        self.use_gpu = torch.cuda.is_available()
        self.batch_size = kwargs.get("batch_size", 20)
        self.max_length = kwargs.get("max_length", 512)
        self.padding = kwargs.get("padding", "longest")
        self.model, self.tokenizer = self._load_model(**kwargs)

    def _load_model(self, **kwargs):
        """
        Loads the pre-trained FiDT5 model and tokenizer.
        
        Parameters
        ----------
        kwargs : dict
            Additional configuration arguments for the model.

        Returns
        -------
        model : transformers.PreTrainedModel
            The loaded model for reranking.                                                                                                                                                                                                                                                                                                                                                                                                       
        tokenizer : transformers.PreTrainedTokenizer
            The tokenizer for the model.
        """
        tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        model = FiDT5.from_pretrained(self.model_name)
        if self.use_gpu:
            model = model.cuda()
        model.eval()
        return model, tokenizer

    def rank(self, documents: list[Document])-> List[Document]:
        """
        Reranks the contexts in each document using the ListT5 model.

        Parameters
        ----------
        documents : list of Document
            A list of Document instances that need to be reranked.

        Returns
        -------
        list of Document
            The reranked list of Document instances.
        """
        for document in documents:
            context_texts = self._prepare_texts(document)
            print(context_texts, len(context_texts))
            input_tensors = self._prepare_inputs(context_texts)
            print(len(input_tensors))
            output = self._run_inference(input_tensors)
            reordered_contexts = self._reorder_contexts(document, output)
            document.reorder_contexts = reordered_contexts
        return documents

    def _prepare_texts(self, document):
        """
        Creates input texts in ListT5 format for a given document.

        Parameters
        ----------
        document : Document
            The document for which to prepare texts.

        Returns
        -------
        list of str
            Formatted texts with context and question information.
        """
        question = document.question.question
        return [f"Query: {question}, Index: {i+1}, Context: {ctx.text}" for i, ctx in enumerate(document.contexts)]

    def _prepare_inputs(self, texts):
        """
        Tokenizes and prepares input tensors for the model.

        Parameters
        ----------
        texts : list of str
            The input texts for each context.

        Returns
        -------
        dict
            Dictionary containing input_ids and attention_mask tensors.
        """
        encoding = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=self.padding,
            max_length=self.max_length,
            truncation=True
        )
        input_tensors = {
            "input_ids": encoding["input_ids"].unsqueeze(0),
            "attention_mask": encoding["attention_mask"].unsqueeze(0)
        }
        if self.use_gpu:
            input_tensors = {key: tensor.cuda() for key, tensor in input_tensors.items()}
        return input_tensors

    def _run_inference(self, input_tensors):
        """
        Runs inference using the ListT5 model to get ranked context scores.

        Parameters
        ----------
        input_tensors : dict
            Tensors containing the input_ids and attention_mask for the model.

        Returns
        -------
        torch.Tensor
            Output tensor with the rankings for each context.
        """
        num_contexts = input_tensors["input_ids"].size(1)
        max_length = num_contexts + 2  # Allow for all contexts to be ranked

        output = self.model.generate(
            **input_tensors,
            max_length=max_length,
            num_beams=num_contexts,  # Match or exceed the total contexts
            return_dict_in_generate=True,
            output_scores=True
        )
        return output

    def _reorder_contexts(self, document, output):
        """
        Reorders contexts in a document based on model output.

        Parameters
        ----------
        document : Document
            The document with contexts to reorder.
        output : torch.Tensor
            Model output scores to determine the order.

        Returns
        -------
        list of Context
            Reordered list of contexts.
        """
        num_contexts = len(document.contexts)
        generated_texts = self.tokenizer.batch_decode(output.sequences, skip_special_tokens=True)
        ranked_indices = [int(idx) - 1 for idx in generated_texts[0].split() if idx.isdigit()]
        # Ensure we include all contexts in the output
        ranked_indices = ranked_indices[:num_contexts]  # Limit to number of contexts if excess
        return [document.contexts[idx] for idx in ranked_indices]