from abc import ABC, abstractmethod
from rankify.dataset.dataset import Document
from typing import List

class BaseGenerator(ABC):
    """
    Abstract base class for **text generation models**.

    This class serves as a blueprint for various **generative models** (e.g., **FiD, T5, GPT**),
    ensuring they implement the `.generate()` method for **answer generation**.

    Attributes
    ----------
    method : str
        The type of generator (`"fid"`, `"t5"`, `"gpt"`, etc.).
    model_name : str
        The name of the specific model (e.g., `"nq_reader_base"` for FiD, `"t5-large"`, `"gpt-4"`).

    """

    def __init__(self, method: str, model_name: str, **kwargs):
        """
        Initializes the **base generator model**.

        Parameters
        ----------
        method : str
            The type of generator (e.g., `"fid"`, `"t5"`, `"gpt"`).
        model_name : str
            The specific model name (e.g., `"nq_reader_base"`, `"t5-large"`, `"gpt-4"`).
        """
        self.method = method
        self.model_name = model_name

    @abstractmethod
    def generate(self, documents: List[Document]) -> List[str]:
        """
        Abstract method for **answer generation** based on input **documents**.

        Parameters
        ----------
        documents : List[Document]
            A list of **Document objects** containing queries and retrieved contexts.

        Returns
        -------
        List[str]
            A list of generated **answers**.
        """
        pass
