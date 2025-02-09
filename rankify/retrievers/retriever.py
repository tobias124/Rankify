
from rankify.dataset.dataset import Document, Question, Answer, Context
from rankify.utils.pre_defined_methods_retrievers import METHOD_MAP

from typing import List

class Retriever:
    def __init__(self, method: str = None, n_docs: int = 10, **kwargs):
        self.method = method
        self.n_docs = n_docs
        self.kwargs = kwargs
        self.retriever = self.initialize()

    def initialize(self):
        if self.method in METHOD_MAP:
            return METHOD_MAP[self.method](n_docs=self.n_docs, **self.kwargs)
        else:
            raise ValueError(f"Retrieving method {self.method} is not supported.")

    def retrieve(self, documents: List[Document]) -> List[Document]:
        """
        Retrieve documents based on a list of questions and optional answers.

        Parameters
        ----------
        questions : List[Question]
            A list of Question objects containing the queries.
        answers : List[Answer], optional
            A list of Answer objects corresponding to the questions.

        Returns
        -------
        List[Document]
            A list of Document objects containing the questions, retrieved contexts, and answers.
        """
        return self.retriever.retrieve(documents)
