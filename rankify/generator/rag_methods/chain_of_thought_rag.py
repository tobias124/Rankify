from typing import List
from rankify.dataset.dataset import Document
from rankify.generator.models.base_rag_model import BaseRAGModel
from rankify.generator.rag_methods.base_rag_method import BaseRAGMethod


class ChainOfThoughtRAG(BaseRAGMethod):
    def __init__(self, model: BaseRAGModel, **kwargs):
        self.model = model

    def answer_questions(self, documents: List[Document], **kwargs) -> List[Document]:
        """Answer a question using chain-of-thought reasoning."""
        answers = []

        for document in documents:
            # Extract question and contexts from the document
            question = document.question.question
            contexts = [context.text for context in document.contexts]

            # Construct the prompt
            prompt = f"""Answer this question using internal chain of thought reasoning, think and
          lay out your logic in multiple steps. You may use the provided contexts, but you can also discard it and just 
             reason by your own knowledge.   :\nQuestion: {question}\nContexts:\n""".join(contexts)
        
            # Generate the answer using the model
            answer = self.model.generate(prompt=prompt, **kwargs)
            
            # Append the answer to the list
            answers.append(answer)
        return answers