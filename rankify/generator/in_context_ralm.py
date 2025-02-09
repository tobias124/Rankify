import os
import json
import torch
from tqdm import tqdm
from rankify.generator.base import BaseGenerator
from rankify.dataset.dataset import Document
from rankify.utils.generator.InContextRalm.model_utils import load_model_and_tokenizer

class InContextRALMGenerator(BaseGenerator):
    """
    In-Context Retrieval-Augmented Language Model (RALM) Generator `[2]`_  for Open-Domain Question Answering.

    .. _[2]: https://arxiv.org/abs/2302.00083

    This class implements **In-Context Retrieval-Augmented Language Models (RALM)** for **context-aware generation**.
    It integrates **retrieved passages** into the input prompt for **few-shot learning**.

    Attributes
    ----------
    device : str
        Device used for inference (`'cuda'` or `'cpu'`).
    model_name : str
        Name of the pre-trained RALM model.
    cache_dir : str
        Directory to store the downloaded models.
    num_docs : int
        Number of retrieved contexts to include in the prompt (default: 1).
    max_length : int
        Maximum number of tokens the model can process.
    max_tokens_to_generate : int
        Maximum number of tokens the model generates in response.

    Methods
    -------
    generate(documents: List[Document]) -> List[str]
        Generates answers based on input queries and retrieved contexts.

    Raises
    ------
    ValueError
        If an unsupported model is provided.

    References
    ----------
    .. [2] Ram, Ori, et al. 
      ["In-Context Retrieval-Augmented Language Models"](https://arxiv.org/abs/2302.00083).
      *Transactions of the Association for Computational Linguistics, 11 (2023): 1316-1331.*

    Examples
    --------
    **Basic Usage:**
    
    >>> from rankify.generator.generator import Generator
    >>> from rankify.dataset.dataset import Document, Question, Answer, Context
    >>> generator = Generator(method="in-context-ralm", model_name="meta-llama/Llama-3.1-8B")
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

    CACHE_DIR = os.environ.get("RERANKING_CACHE_DIR", "./cache")

    def __init__(self, method: str = "in-context-ralm", model_name: str = "meta-llama/Llama-3.1-8B", **kwargs):
        """
        Initializes the RALM Generator.
        """
        super().__init__(method, model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.cache_dir = self.CACHE_DIR
        self.num_docs = kwargs.get("num_docs", 1)  # Default: 1 supporting document

        # Load the model and tokenizer
        self.model, self.tokenizer, self.config, self.device = load_model_and_tokenizer(
            self.model_name
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Set generation parameters
        self.max_length = self.config.n_positions if hasattr(self.config, "n_positions") else self.config.max_position_embeddings
        self.max_tokens_to_generate = kwargs.get("max_tokens_to_generate", 10)

    def _build_qa_prompt(self, example):
        """
        Constructs the QA prompt for in-context learning.
        """
        question = example["question"]
        question = question[0].lower() + question[1:] if not question.endswith("?") else question

        if self.num_docs == 0:
            return f"Answer these questions:\nQ: {question}\nA:"
        
        docs_text = "\n\n".join([f"{ctx['title']}\n\n{ctx['text']}" for ctx in example["ctxs"][:self.num_docs]])
        return f"{docs_text}\n\nBased on these texts, answer these questions:\nQ: {question}\nA:"

    def _prepare_dataloader(self, documents: list[Document]):
        """
        Converts `Document` objects into a RALM-compatible dataset.
        """
        examples = []
        for doc in documents:
            example = {
                "question": doc.question.question,
                "answers": doc.answers.answers,
                "ctxs": [{"title": ctx.title, "text": ctx.text, "score": ctx.score} for ctx in doc.contexts]
            }
            examples.append(example)

        return examples

    def generate(self, documents: list[Document]) -> list[str]:
        """
        Generates answers for a list of documents using RALM.
        """
        eval_dataset = self._prepare_dataloader(documents)

        results = []
        for example in tqdm(eval_dataset, desc="Generating responses"):
            prompt = self._build_qa_prompt(example)

            tokenized_input = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            input_ids = tokenized_input.input_ids.to(self.device)
            attention_mask = tokenized_input.attention_mask.to(self.device)  # Extract attention mask

            if input_ids.shape[-1] > self.max_length - self.max_tokens_to_generate:
                input_ids = input_ids[..., -(self.max_length - self.max_tokens_to_generate):]
                attention_mask = attention_mask[..., -(self.max_length - self.max_tokens_to_generate):]

            with torch.no_grad():
                outputs = self.model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=self.max_tokens_to_generate,  pad_token_id=self.tokenizer.pad_token_id )

            # Extract generated text
            generation_str = self.tokenizer.decode(outputs[0].cpu(), skip_special_tokens=True)
            answer = generation_str[len(prompt):].split("\n")[0]

            results.append(answer)

        return results

