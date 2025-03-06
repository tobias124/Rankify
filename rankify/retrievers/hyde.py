from .contriever import ContrieverRetriever
from rankify.utils.retrievers.hyde import Promptor,  OpenAIGenerator
from rankify.dataset.dataset import Document, Context, Question, Answer
from typing import List
from tqdm import tqdm
from pyserini.eval.evaluate_dpr_retrieval import has_answers, SimpleTokenizer

import numpy as np
class HydeRetreiver:

    def __init__(self, model = "facebook/contriever-msmarco", n_docs: int = 100,batch_size: int = 32, device: str = "cuda", index_type: str = "wiki", **kwargs):
        self.task = kwargs.get("task", 'web search')
        self.key= kwargs.get("api_key", None)
        self.llm_model  =kwargs.get("llm_model", 'gpt-3.5-turbo-0125')
        self.base_url  =kwargs.get("base_url", None)
        self.num_generated_docs  =kwargs.get("num_generated_docs", 1)
        self.max_token_generated_docs  =kwargs.get("max_token_generated_docs", 512)
        self.n_docs = n_docs
        self.batch_size = batch_size
        self.promptor  =  Promptor(self.task)
        self.generator = OpenAIGenerator(model_name = self.llm_model , api_key=self.key, base_url=self.base_url, n=self.num_generated_docs, max_tokens=self.max_token_generated_docs, temperature=0.7, top_p=1, frequency_penalty=0.0, presence_penalty=0.0, stop=['\n\n\n'], wait_till_success=False)
        self.contriever = ContrieverRetriever(model, n_docs, batch_size, device, index_type)

        self.searcher = self.contriever.index
    def retrieve(self, documents: List[Document]):
        for i , document in enumerate(tqdm(documents, desc="Processing documents", unit="docs")):
            contexts = []
            prompt = self.promptor.build_prompt(document.question.question.replace("?",""))
            hypothesis_documents = self.generator.generate(prompt)
            
            all_emb_c = []
            for c in [prompt] + hypothesis_documents:
                c_emb = self.contriever._embed_queries([c])
                all_emb_c.append(c_emb.squeeze(0))
            all_emb_c = np.array(all_emb_c)
            avg_emb_c = np.mean(all_emb_c,axis = 0)
            hype_vector = avg_emb_c.reshape((1, len(avg_emb_c)))
            top_ids_and_scores = self.contriever.index.search_knn(hype_vector, self.n_docs, index_batch_size=self.batch_size)
            doc_ids, scores = top_ids_and_scores[0]
            for doc_id, score  in zip( doc_ids, scores):
                try:
                    passage = self.contriever.passage_id_map[int(doc_id)]
                    context = Context(
                        id=int(doc_id),
                        title=passage["title"],
                        text=passage["text"],
                        score=score,
                        has_answer=has_answers(passage["text"], document.answers.answers, SimpleTokenizer(), regex=False)  # Could be updated with a function to check for answers
                    )
                    contexts.append(context)
                except (IndexError, KeyError):
                    # Log or handle the error, and continue with the next passage
                    continue
            document.contexts = contexts
        return documents

                
            





