
import rankify 
import os

from rankify.dataset.dataset import Document, Question, Answer, Context
from rankify.retrievers.retriever import Retriever

# Sample Documents
documents = [
    Document(question=Question("the cast of a good day to die hard?"), answers=Answer([
            "Jai Courtney",
            "Sebastian Koch",
            "Radivoje Bukvi\u0107",
            "Yuliya Snigir",
            "Sergei Kolesnikov",
            "Mary Elizabeth Winstead",
            "Bruce Willis"
        ]), contexts=[]),
    Document(question=Question("Who wrote Hamlet?"), answers=Answer(["Shakespeare"]), contexts=[])
]




"""retriever = Retriever(method="colbert", n_docs=1 , index_type="wiki" )
retrieved_documents = retriever.retrieve(documents)

# Print the first retrieved document
for i, doc in enumerate(retrieved_documents):
    print(f"\nDocument {i+1}:")
    print(doc)

retriever = Retriever(method="colbert", n_docs=1 , index_type="msmarco" )
retrieved_documents = retriever.retrieve(documents)


# Print the first retrieved document
for i, doc in enumerate(retrieved_documents):
    print(f"\nDocument {i+1}:")
    print(doc)
"""

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
retriever = Retriever(method="hyde", n_docs=2 , index_type="wiki", api_key=OPENAI_API_KEY )
retrieved_documents = retriever.retrieve(documents)

# Print the first retrieved document
for i, doc in enumerate(retrieved_documents):
    print(f"\nDocument {i+1}:")
    print(doc)

retriever = Retriever(method="bge", n_docs=1 , index_type="msmarco" )
retrieved_documents = retriever.retrieve(documents)
# Print the first retrieved document
for i, doc in enumerate(retrieved_documents):
    print(f"\nDocument {i+1}:")
    print(doc)



