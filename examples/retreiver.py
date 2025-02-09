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

retriever = Retriever(method="bge", n_docs=5 , index_type="wiki" )
retrieved_documents = retriever.retrieve(documents)

# Print the first retrieved document
for i, doc in enumerate(retrieved_documents):
    print(f"\nDocument {i+1}:")
    print(doc)
