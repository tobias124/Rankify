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
    **FiD (Fusion-in-Decoder) Generator** for Open-Domain Question Answering.


    **Fusion-in-Decoder (FiD)** is a **retrieval-augmented generation (RAG) model** that 
    aggregates information from **multiple retrieved passages** to generate context-aware answers.

    Attributes:
        device (str): Device used for inference (`'cuda'` or `'cpu'`).
        model_path (str): Path to the downloaded and extracted model.
        tokenizer (transformers.T5Tokenizer): T5 tokenizer for text encoding.
        model (FiDT5): Pretrained FiD model for text generation.
        max_length (int): Maximum length of the generated output (default: 50 tokens).

    References:
        - **Izacard & Grave** *Leveraging Passage Retrieval with Generative Models for Open-Domain QA*  
          [Paper](https://arxiv.org/abs/2007.01282)

    See Also:
        - `BaseGenerator`: Parent class for FiDGenerator.
        - `RAG-based QA Models`: FiD falls under retrieval-augmented generation.

    Example:
        ```python
        from rankify.generator.generator import Generator
        from rankify.dataset.dataset import Document, Question, Answer, Context

        generator = Generator(method="fid", model_name="nq_reader_base")
        documents = [
            Document(
                question=Question("Who discovered gravity?"),
                answers=Answer(["Isaac Newton"]),
                contexts=[
                    Context(title="Gravity", text="Isaac Newton formulated the laws of gravity.", score=1.0)
                ]
            )
        ]

        generated_answers = generator.generate(documents)
        print(generated_answers[0])  # 'Isaac Newton discovered gravity in the late 17th century.'
        ```

    Notes:
        - FiD **combines multiple passages** to generate better responses.
        - It integrates **seamlessly** with the `Generator` class.
        - Uses **retrieval-augmented generation (RAG)** techniques for QA.
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

        Args:
            method (str): The generator type (`"fid"`).
            model_name (str): The specific FiD model to use (e.g., `"nq_reader_base"`).
            **kwargs: Additional parameters for model configuration.

        Example:
            ```python
            generator = FiDGenerator(method="fid", model_name="nq_reader_base")
            ```
        """
        super().__init__(method, model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = self._download_model()
        self.tokenizer, self.model = self._load_model()
        self.max_length = kwargs.get("max_length", 50)  # Default max length for answers

    def _download_model(self) -> str:
        """
        Downloads and extracts the FiD model if not already present.

        Returns:
            str: The directory path where the model is stored.

        Raises:
            ValueError: If an unknown FiD model is specified.

        Example:
            ```python
            model_path = generator._download_model()
            ```
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

        Returns:
            tuple: A tuple containing the tokenizer and the FiD model.

        Example:
            ```python
            tokenizer, model = generator._load_model()
            ```
        """
        tokenizer = transformers.T5Tokenizer.from_pretrained('t5-base', return_dict=False)
        model = FiDT5.from_pretrained(self.model_path)
        model = model.to(self.device)
        model.eval()
        return tokenizer, model

    def _prepare_dataloader(self, documents: list[Document]):
        """
        Converts `Document` objects into a FiD-compatible dataset.

        Args:
            documents (List[Document]): A list of documents with queries and retrieved contexts.

        Returns:
            tuple: A tuple containing a DataLoader and Dataset.

        Example:
            ```python
            dataloader, dataset = generator._prepare_dataloader(documents)
            ```
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

        Args:
            documents (List[Document]): A list of documents with queries and retrieved contexts.

        Returns:
            List[str]: A list of generated answers.

        Example:
            ```python
            generator = FiDGenerator(method="fid", model_name="nq_reader_base")
            documents = [Document(question=Question("Who wrote Hamlet?"))]
            print(generator.generate(documents))  # ['William Shakespeare wrote Hamlet in the early 1600s.']
            ```
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
