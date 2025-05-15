from typing import List
from rankify.dataset.dataset import Document
from rankify.generator.base_rag_model import BaseRAGModel
from rankify.generator.rag_methods.base_rag_method import BaseRAGMethod


class ChainOfThoughtRAG(BaseRAGMethod):
    def __init__(self, model: BaseRAGModel):
        self.model = model

    def answer_question(self, documents: List[Document], **kwargs) -> str:
        """Answer a question using chain-of-thought reasoning."""
        
        # Extract question from the first document
        question = documents[0].question.question
        # Extract contexts from all documents
        contexts = []
        for document in documents:
            contexts.extend([context.text for context in document.contexts])
        
        prompt = f"""Answer this question using internal chain of thought reasoning, think and
          lay out your logic in multiple steps. You may use the provided contexts, but you can also discard it and just 
             reason by your own knowledge.   :\nQuestion: {question}\nContexts:\n""".join(contexts)
        return self.model.generate(prompt, **kwargs)