import asyncio
import os
import subprocess
from typing import List, TypeVar
import tempfile
from loguru import logger
from pyserini.eval.evaluate_dpr_retrieval import has_answers, SimpleTokenizer
from tqdm import tqdm

from rankify.dataset.dataset import Document, Question, Context, Answer
from rankify.tools.Tools import WebSearchTool
import json
from pyserini.search.lucene import LuceneSearcher


T = TypeVar('T')

class OnlineRetriever:

    """
    Implements Online Retrieval technique to retrieve relevant documents from the web.
    Implements a web search retrieval using an AI agent s.t. CodeAgent or ReAct Agent.
    References:
        - https://huggingface.co/docs/smolagents/en/index
        - https://github.com/unclecode/crawl4ai
        - https://www.litellm.ai/

      Attributes:
        n_docs (int): Number of top documents to retrieve per query.
        batch_size (int): Number of queries processed in a single batch for efficiency.
        searcher (LuceneSearcher): Pyserini-based search engine for BM25 retrieval.
        llm_model (str): name of LLM model to be used in Crawl4AI scraping.

    Example:
        ```python
        from rankify.dataset.dataset import Document, Question
        from rankify.retrievers.retriever import Retriever

         documents = [
            Document(question=Question("who cofounded apple inc, alongside steve jobs?"), answers=Answer([
                "Steve Wozniak ",
                "Ronald Wayne"
            ]), contexts=[])
        ]

        online_retriever = Retriever(method="online_retriever")
        retrieved_documents = online_retriever.retrieve(documents)

        for i, doc in enumerate(retrieved_documents):
            print(f"\nDocument {i + 1}:")
            print(doc)

    """

    def __init__(self,model:str='online_retriever',n_docs:int=10,llm_model:str=None,batch_size:int = 10)->None:
        self.n_docs = n_docs
        self.searcher = WebSearchTool()
        self.model = model
        self.llm_model = llm_model
        self.sources = []
        self.relevant_docs:List[Document]=[]
        self.batch_size = batch_size
        self.tokenizer = SimpleTokenizer()



    def _search_web(self,query:str)->List[T]:
        if not self.searcher.is_initialized:
            self.searcher.setup()
        return self.searcher.forward(query,num_result=self.n_docs)


    def retrieve(self,documents:List[Document]) -> List[Document]:
        question_texts =[doc.question.question for doc in documents]

        for i,question in enumerate(tqdm(question_texts,desc="Fetching documents...")):
            document = documents[i]
            logger.info(f"🌐 Retrieving contexts for q{i}:{question}...")
            sources = self._search_web(question)
            # Prepare it to be indexed by Pyserini
            with tempfile.TemporaryDirectory() as tmpdirname:
                # index the context to be searched by LuceneSearcher
                contexts = [{'id': i + 1, 'contents': '\n'.join([doc['title'], doc['link'], doc['snippet'], doc['html']])} for i, doc in enumerate(sources)]
                json.dump(contexts, open(tmpdirname + '/context.json', 'w'))
                subprocess.run([
                    "python", "-m", "pyserini.index.lucene",
                    "-collection", "JsonCollection",
                    "-generator", "DefaultLuceneDocumentGenerator",
                    "-input", tmpdirname,
                    "-index", tmpdirname,
                    "-storePositions", "-storeDocvectors", "-storeRaw"
                ], check=True)

                # Search the indexed contexts for similarity with question
                lucene_searcher = LuceneSearcher(tmpdirname)
                hits = lucene_searcher.search(question)

                # Build the context of the document
                contexts:List[Context]= []
                for hit in hits:
                    try:
                        lucene_doc = lucene_searcher.doc(hit.docid)
                        raw_content = json.loads(lucene_doc.raw())
                        text = raw_content.get("contents","")
                        title = raw_content.get("contents","").split("\n")[0]
                        context = Context(
                            id=int(hit.docid),
                            title=title,
                            text=text,
                            score=hit.score,
                            has_answer=has_answers(text, document.answers.answers, self.tokenizer)
                        )
                        contexts.append(context)
                    except Exception as e:
                        print(f"Error processing document ID {hit.docid}: {e}")
                # Assign the retrieved contexts to the document
                document.contexts = contexts

        return documents





