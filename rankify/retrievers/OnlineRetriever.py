import asyncio
import os
import subprocess
from typing import List, TypeVar, Dict, Any
import tempfile
from loguru import logger
from pyserini.eval.evaluate_dpr_retrieval import has_answers, SimpleTokenizer
from tqdm import tqdm

from rankify.dataset.dataset import Document, Question, Context, Answer
from rankify.tools.Tools import WebSearchTool
import json
from pyserini.search.lucene import LuceneSearcher

import re
import html2text
from bs4 import BeautifulSoup
from typing import List
T = TypeVar('T')


import html2text
from bs4 import BeautifulSoup
import re
from typing import List, Dict, Any

class WikiProcessor:
    def __init__(self):
        # Initialize html2text
        self.h2t = html2text.HTML2Text()
        self.h2t.ignore_links = True  # Remove hyperlinks
        self.h2t.ignore_images = True  # Remove image links
        self.h2t.ignore_tables = True  # Skip tables
        self.h2t.body_width = 0  # Disable line wrapping

    def extract_title_from_html(self, html_content: str) -> str:
        """Extract the title from HTML content."""
        soup = BeautifulSoup(html_content, 'lxml')
        title_tag = soup.find('title')
        return title_tag.text.strip() if title_tag else "No Title"

    def clean_html_content(self, html_content: str) -> str:
        """Clean HTML content and convert to plain text."""
        # Remove scripts, styles, and other unwanted tags
        soup = BeautifulSoup(html_content, 'lxml')
        for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
            tag.decompose()

        # Convert to text using html2text
        clean_text = self.h2t.handle(str(soup))

        # Remove extra whitespace and newlines
        clean_text = re.sub(r'\n\s*\n+', '\n\n', clean_text.strip())
        return clean_text

    def split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        # Split by double newlines (html2text preserves paragraphs as double newlines)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        # Filter out very short paragraphs or unwanted content (e.g., navigation text)
        return [p for p in paragraphs if len(p) > 10]

    def process_wikipedia_content(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process Wikipedia sources and split into paragraphs."""
        processed_contexts = []
        
        for i, source in enumerate(sources):
            html_content = source.get('html', '')
            if not html_content:
                continue
            
            # Extract title
            title = self.extract_title_from_html(html_content)
            
            # Clean HTML and extract text
            clean_text = self.clean_html_content(html_content)
            
            # Split into paragraphs
            paragraphs = self.split_into_paragraphs(clean_text)
            
            logger.info(f"Extracted {len(paragraphs)} paragraphs from {title}")
            
            # Create separate context for each paragraph
            for j, paragraph in enumerate(paragraphs):
                context_id = f"{i}_{j}"
                processed_contexts.append({
                    'id': context_id,
                    'contents': paragraph,
                    'title': title,
                    'paragraph_index': j,
                    'source_url': source.get('link', '') or source.get('url', ''),
                    'original_source_index': i
                })
        
        return processed_contexts



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

    def __init__(self,model:str='online_retriever',n_docs:int=10,llm_model:str=None,batch_size:int = 10, api_key=None)->None:
        self.n_docs = n_docs
        print(api_key)
        self.searcher = WebSearchTool(search_provider_api_key=api_key)
        self.model = model
        self.llm_model = llm_model
        self.sources = []
        self.relevant_docs:List[Document]=[]
        self.batch_size = batch_size
        self.tokenizer = SimpleTokenizer()
        self.wiki_processor = WikiProcessor()  # Initialize WikiProcessor


    def _search_web(self,query:str)->List[T]:
        if not self.searcher.is_initialized:
            self.searcher.setup()
        return self.searcher.forward(query,num_result=self.n_docs)

    def _process_wikipedia_content(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return self.wiki_processor.process_wikipedia_content(sources)
    def retrieve(self,documents:List[Document]) -> List[Document]:
        question_texts =[doc.question.question for doc in documents]

        for i,question in enumerate(tqdm(question_texts,desc="Fetching documents...")):
            document = documents[i]
            logger.info(f"🌐 Retrieving contexts for q{i}:{question}...")
            sources = self._search_web(question)
            print(f"Retrieved {sources} ")
            # Prepare it to be indexed by Pyserini
            processed_contexts = self._process_wikipedia_content(sources)
            with tempfile.TemporaryDirectory() as tmpdirname:
                # index the context to be searched by LuceneSearcher
                #contexts = [{'id': i + 1, 'contents': '\n'.join([doc['title'], doc['link'], doc['snippet'], doc['html']])} for i, doc in enumerate(sources)]
                
                # contexts = [{
                #     'id': i + 1,
                #     'contents': '\n'.join([
                #         #str(doc.get('title') or ''),
                #         #str(doc.get('link') or ''),
                #         #str(doc.get('snippet') or ''),
                #         str(doc.get('html') or '')
                #     ])
                # } for i, doc in enumerate(sources)]
                contexts = [
                    {
                        'id': ctx['id'],
                        'contents': '\n'.join([ctx['title'], ctx['contents'], ctx['source_url']])
                    }
                    for ctx in processed_contexts
                ]
                
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





