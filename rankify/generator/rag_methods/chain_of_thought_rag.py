from typing import List
from rankify.generator.base_rag_model import BaseRAGModel


class ChainOfThoughtRAG:
    def __init__(self, model: BaseRAGModel):
        self.model = model

    def answer_question(self, question: str, contexts: List[str], **kwargs) -> str:
        """Answer a question using chain-of-thought reasoning."""
        prompt = f"Chain of Thought:\nQuestion: {question}\nContexts:\n" + "\n".join(contexts)
        return self.model.generate(prompt, **kwargs)