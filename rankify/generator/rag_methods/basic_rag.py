from typing import List
from rankify.generator.base_rag_model import BaseRAGModel

from typing import List
from rankify.dataset.dataset import Document

class BasicRAG:
    def __init__(self, model: BaseRAGModel):
        self.model = model

    def answer_question(self, documents: List[Document]) -> List[str]:
        """
        Answer questions for a list of documents using the model.

        Args:
            documents (List[Document]): A list of Document objects containing questions and contexts.

        Returns:
            List[str]: A list of answers, one for each document.
        """
        answers = []
        for document in documents:
            # Extract question and contexts from the document
            question = document.question.question
            contexts = [context.text for context in document.contexts]

            # Construct the prompt
            prompt = f"Question: {question}\nContexts:\n" + "\n".join(contexts)

            # Generate the answer using the model
            answer = self.model.generate(prompt, max_new_tokens=50)
            answers.append(answer)

        return answers