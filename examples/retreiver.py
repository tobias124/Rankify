import rankify 
import os

from rankify.dataset.dataset import Document, Question, Answer, Context
from rankify.retrievers.retriever import Retriever

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
    # Document(question=Question("Who wrote Hamlet?"), answers=Answer(["Shakespeare"]), contexts=[])
]

retriever = Retriever(method='ance-multi', n_docs=1, index_type='msmarco')
retrieved_documents = retriever.retrieve(documents)
for i, doc in enumerate(retrieved_documents):
    print(f"\nDocument {i+1}:")
    print(doc)


retriever = Retriever(method='ance-multi', n_docs=1, index_type='wiki')
retrieved_documents = retriever.retrieve(documents)
for i, doc in enumerate(retrieved_documents):
    print(f"\nDocument {i+1}:")
    print(doc)


retriever = Retriever(method='dpr-single', n_docs=1, index_type='wiki')
retrieved_documents = retriever.retrieve(documents)
for i, doc in enumerate(retrieved_documents):
    print(f"\nDocument {i+1}:")
    print(doc)


retriever = Retriever(method='dpr-single', n_docs=1, index_type='msmarco')
retrieved_documents = retriever.retrieve(documents)
for i, doc in enumerate(retrieved_documents):
     print(f"\nDocument {i+1}:")
     print(doc)


retriever = Retriever(method='dpr-multi', n_docs=1, index_type='wiki')
retrieved_documents = retriever.retrieve(documents)
for i, doc in enumerate(retrieved_documents):
    print(f"\nDocument {i+1}:")
    print(doc)


retriever = Retriever(method='dpr-multi', n_docs=1, index_type='msmarco')
retrieved_documents = retriever.retrieve(documents)
for i, doc in enumerate(retrieved_documents):
     print(f"\nDocument {i+1}:")
     print(doc)





retriever = Retriever(method='bm25', n_docs=1, index_type='msmarco')
retrieved_documents = retriever.retrieve(documents)
for i, doc in enumerate(retrieved_documents):
    print(f"\nDocument {i+1}:")
    print(doc)


retriever = Retriever(method='bm25', n_docs=1, index_type='wiki')
retrieved_documents = retriever.retrieve(documents)
for i, doc in enumerate(retrieved_documents):
    print(f"\nDocument {i+1}:")
    print(doc)




retriever = Retriever(method='bge', n_docs=1, index_type='wiki')
retrieved_documents = retriever.retrieve(documents)
for i, doc in enumerate(retrieved_documents):
    print(f"\nDocument {i+1}:")
    print(doc)

retriever = Retriever(method='bge', n_docs=1, index_type='msmarco')
retrieved_documents = retriever.retrieve(documents)
for i, doc in enumerate(retrieved_documents):
    print(f"\nDocument {i+1}:")
    print(doc)



retriever = Retriever(method='colbert', n_docs=1, index_type='wiki')
retrieved_documents = retriever.retrieve(documents)
for i, doc in enumerate(retrieved_documents):
    print(f"\nDocument {i+1}:")
    print(doc)

retriever = Retriever(method='colbert', n_docs=1, index_type='msmarco')
retrieved_documents = retriever.retrieve(documents)
for i, doc in enumerate(retrieved_documents):
    print(f"\nDocument {i+1}:")
    print(doc)



retriever = Retriever(method='contriever', n_docs=1, index_type='wiki')
retrieved_documents = retriever.retrieve(documents)
for i, doc in enumerate(retrieved_documents):
    print(f"\nDocument {i+1}:")
    print(doc)

retriever = Retriever(method='contriever', n_docs=1, index_type='msmarco', model="facebook/contriever-msmarco")
retrieved_documents = retriever.retrieve(documents)
for i, doc in enumerate(retrieved_documents):
    print(f"\nDocument {i+1}:")
    print(doc)






retriever = Retriever(method='bm25', n_docs=5, index_folder='./indices/EU_corpus')
retrieved_documents = retriever.retrieve(documents)
for i, doc in enumerate(retrieved_documents):
     print(f"\nDocument {i+1}:")
     print(doc)



retriever = Retriever(method='contriever', n_docs=5, index_folder='./indices/EU_corpus/contriever_index_wiki')
retrieved_documents = retriever.retrieve(documents)
for i, doc in enumerate(retrieved_documents):
     print(f"\nDocument {i+1}:")
     print(doc)


retriever = Retriever(method='bge', n_docs=5, index_folder='./indices/EU_corpus/bge_index_wiki')
retrieved_documents = retriever.retrieve(documents)
for i, doc in enumerate(retrieved_documents):
    print(f"\nDocument {i+1}:")
    print(doc)

# for custom index you need tsv file call passages.tsv
retriever = Retriever(method='colbert', n_docs=5, index_folder='/gpfs/gpfs1/scratch/c7031420/rankify/unit_test/indices/EU_corpus/colbert_index_wiki')
retrieved_documents = retriever.retrieve(documents)
for i, doc in enumerate(retrieved_documents):
    print(f"\nDocument {i+1}:")
    print(doc)

retriever = Retriever(method='dpr-single', n_docs=5, index_folder='./indices/EU_corpus/dpr_index_wiki')
retrieved_documents = retriever.retrieve(documents)
for i, doc in enumerate(retrieved_documents):
    print(f"\nDocument {i+1}:")
    print(doc)




serper = "" 
retriever = Retriever(method='online', n_docs=5,   api_key=serper)
retrieved_documents = retriever.retrieve(documents)
for i, doc in enumerate(retrieved_documents):
    print(f"\nDocument {i+1}:")
    print(doc)



OPENAI_API_KEY = ""
retriever = Retriever(method='hyde', n_docs=5, index_type='wiki',   api_key=OPENAI_API_KEY)
retrieved_documents = retriever.retrieve(documents)
for i, doc in enumerate(retrieved_documents):
    print(f"\nDocument {i+1}:")
    print(doc)