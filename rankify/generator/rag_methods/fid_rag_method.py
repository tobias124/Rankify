from typing import List
from rankify.generator.models.base_rag_model import BaseRAGModel

from typing import List
from rankify.dataset.dataset import Document
from rankify.generator.rag_methods.base_rag_method import BaseRAGMethod

class FiDRAGMethod(BaseRAGMethod):
    def __init__(self, model: BaseRAGModel):
        self.model = model

    def answer_questions(self, documents: List[Document], **kwargs) -> List[str]:
        """
        Answer question for a list of documents using the model.

        Args:
            documents (List[Document]): A list of Document objects containing questions and contexts.

        Returns:
            str: An answer based on the given documents and question.
        """

        # Generate the answer using the model
        answer = self.model.generate(documents, **kwargs)
        
        return answer