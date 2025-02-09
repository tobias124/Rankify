import os
import torch
import json
from pathlib import Path
from torch.utils.data import DataLoader, SequentialSampler
from rankify.utils.generator.FiD.data import Dataset, Collator
#from rankify.utils.generator.FiD.util import load_t5_tokenizer
from  rankify.utils.generator.FiD.model import FiDT5
from rankify.generator.base import BaseGenerator
from rankify.utils.generator.download import ModelDownloader
from rankify.dataset.dataset import Document
import transformers
class FiDGenerator(BaseGenerator):
    """
    **FiD (Fusion-in-Decoder) Generator `[1]`_ ** for Open-Domain Question Answering.

    .. _[1]: https://arxiv.org/abs/2007.01282

    This class implements **Fusion-in-Decoder (FiD) model** for **retrieval-augmented generation (RAG)**.
    It combines **multiple retrieved passages** to generate **context-aware answers**.

    Attributes
    ----------
    device : str
        Device used for inference (`'cuda'` or `'cpu'`).
    model_path : str
        Path to the downloaded and extracted model.
    tokenizer : transformers.T5Tokenizer
        T5 tokenizer for text encoding.
    model : FiDT5
        Pretrained FiD model for text generation.
    max_length : int
        Maximum length of the generated output (default: 50 tokens).

    Methods
    -------
    generate(documents: List[Document]) -> List[str]
        Generates answers for input queries using retrieved passages.

    Raises
    ------
    ValueError
        If an unsupported model name is provided.

    References
    ----------
    .. [1] Izacard, Gautier, and Edouard Grave. 
      ["Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering"](https://arxiv.org/abs/2007.01282).
      *arXiv preprint arXiv:2007.01282 (2020).*

    Examples
    --------
    **Basic Usage:**
    
    >>> from rankify.generator.generator import Generator
    >>> from rankify.dataset.dataset import Document, Question, Answer, Context
    >>> generator = Generator(method="fid", model_name="nq_reader_base")
    >>> documents = [
    ...     Document(
    ...         question=Question("Who discovered gravity?"),
    ...         answers=Answer(["Isaac Newton"]),
    ...         contexts=[
    ...             Context(title="Gravity", text="Isaac Newton formulated the laws of gravity.", score=1.0)
    ...         ]
    ...     )
    ... ]
    >>> generated_answers = generator.generate(documents)
    >>> print(generated_answers[0])
    'Isaac Newton discovered gravity in the late 17th century.'
    """

    MODEL_URLS = {
        "nq_reader_base": "https://dl.fbaipublicfiles.com/FiD/pretrained_models/nq_reader_base.tar.gz",
        "nq_reader_large": "https://dl.fbaipublicfiles.com/FiD/pretrained_models/nq_reader_large.tar.gz",
        "tqa_reader_base": "https://dl.fbaipublicfiles.com/FiD/pretrained_models/tqa_reader_base.tar.gz",
        "tqa_reader_large": "https://dl.fbaipublicfiles.com/FiD/pretrained_models/tqa_reader_large.tar.gz"
    }

    CACHE_DIR = os.environ.get("RERANKING_CACHE_DIR", "./cache")

    def __init__(self, method: str = "fid", model_name: str = "nq_reader_base", **kwargs):
        """
        Initializes the FiD Generator.

        Parameters
        ----------
        method : str
            The generator type (`"fid"`).
        model_name : str
            The specific FiD model to use (e.g., `"nq_reader_base"`).
        kwargs : dict
            Additional parameters for model configuration.
        """
        super().__init__(method, model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = self._download_model()
        self.tokenizer, self.model = self._load_model()
        self.max_length = kwargs.get("max_length", 50)  # Default max length for answers

    def _download_model(self) -> str:
        """
        Downloads and extracts the FiD model if not already present.

        Returns
        -------
        str
            The directory path where the model is stored.

        Raises
        ------
        ValueError
            If an unknown FiD model is specified.
        """
        model_url = self.MODEL_URLS.get(self.model_name)
        if not model_url:
            raise ValueError(f"Unknown FiD model: {self.model_name}")

        model_dir = os.path.join(self.CACHE_DIR, "generator", "fid", self.model_name)
        if not os.path.exists(model_dir):
            ModelDownloader.download_and_extract(model_url, model_dir)

        return model_dir

    def _load_model(self):
        """
        Loads the FiD model and tokenizer.

        Returns
        -------
        tuple
            A tuple containing the tokenizer and the FiD model.
        """
        tokenizer = transformers.T5Tokenizer.from_pretrained('t5-base', return_dict=False)
        model = FiDT5.from_pretrained(self.model_path)
        model = model.to(self.device)
        model.eval()
        return tokenizer, model

    def _prepare_dataloader(self, documents: list[Document]):
        """
        Converts `Document` objects into a FiD-compatible dataset.

        Parameters
        ----------
        documents : List[Document]
            A list of documents with queries and retrieved contexts.

        Returns
        -------
        tuple
            A tuple containing a DataLoader and Dataset.
        """
        examples = []
        for doc in documents:
            example = {
                "question": doc.question.question,
                "answers": doc.answers.answers,
                "ctxs": [{"title": ctx.title, "text": ctx.text, "score": ctx.score} for ctx in doc.contexts]
            }
            examples.append(example)

        eval_dataset = Dataset(examples, n_context=len(documents[0].contexts))
        eval_sampler = SequentialSampler(eval_dataset)
        collator_function = Collator(512, self.tokenizer)  # 512 is max text length

        eval_dataloader = DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=4,  # Modify as needed
            num_workers=4,
            collate_fn=collator_function
        )

        return eval_dataloader, eval_dataset

    def generate(self, documents: list[Document]) -> list[str]:
        """
        Generates answers for a list of documents.

        Parameters
        ----------
        documents : List[Document]
            A list of documents with queries and retrieved contexts.

        Returns
        -------
        List[str]
            A list of generated answers.

        Example
        -------
        >>> generator = FiDGenerator(method="fid", model_name="nq_reader_base")
        >>> documents = [Document(question=Question("Who wrote Hamlet?"))]
        >>> print(generator.generate(documents))
        ['William Shakespeare wrote Hamlet in the early 1600s.']
        """
        eval_dataloader, eval_dataset = self._prepare_dataloader(documents)

        results = []
        with torch.no_grad():
            for batch in eval_dataloader:
                (idx, _, _, context_ids, context_mask) = batch

                outputs = self.model.generate(
                    input_ids=context_ids.to(self.device),
                    attention_mask=context_mask.to(self.device),
                    max_length=self.max_length
                )

                for k, output in enumerate(outputs):
                    answer = self.tokenizer.decode(output, skip_special_tokens=True)
                    results.append(answer)

        return results
