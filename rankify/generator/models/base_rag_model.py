from abc import ABC, abstractmethod
from typing import List

class BaseRAGModel(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response based on the given prompt."""
        pass

    def embed(self, text: str, **kwargs) -> List[float]:
        """Optional: Generate embeddings for the given text."""
        raise NotImplementedError("Embedding is not required for this implementation.")