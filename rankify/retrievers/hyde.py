from .contriever import ContrieverRetriever
from rankify.utils.retrievers.hyde import Promptor,  OpenAIGenerator
from rankify.dataset.dataset import Document, Context, Question, Answer
from typing import List
from tqdm import tqdm
from pyserini.eval.evaluate_dpr_retrieval import has_answers, SimpleTokenizer

import numpy as np
class HydeRetreiver:
    """
    Implements **Hypothetical Document Embedding (HyDE) Retriever**, 
    which enhances document retrieval by generating hypothetical documents 
    using an LLM and averaging their embeddings for improved search performance.

    HyDE Retriever leverages a **large language model (LLM)** to generate 
    potential answers (hypothetical documents) based on a query. These generated 
    texts are embedded and combined with the query embedding to retrieve 
    relevant documents using FAISS-based search.

    References:
        - **Luyu Gao,Xueguang Ma, Jimmy Lin, and Jamie Callan. (2022)**: *Precise Zero-Shot Dense Retrieval without Relevance Labels*.
          [Paper](https://arxiv.org/abs/2212.10496)

    Attributes:
        task (str): The task type for the **prompt generator** (e.g., `"web search"`).
        key (str, optional): API key for the **LLM generator** (default: `None`).
        llm_model (str): The LLM model used for hypothetical document generation (default: `"gpt-3.5-turbo-0125"`).
        base_url (str, optional): Custom API base URL for OpenAI requests.
        num_generated_docs (int): Number of **hypothetical documents** generated per query.
        max_token_generated_docs (int): Maximum number of **tokens** in generated documents.
        n_docs (int): Number of **top documents** to retrieve per query.
        batch_size (int): Number of **queries processed per batch**.
        promptor (Promptor): Instance of the **Promptor class** for generating prompts.
        generator (OpenAIGenerator): Instance of **OpenAIGenerator** for generating hypothetical documents.
        contriever (ContrieverRetriever): The **dense retrieval model** used for passage retrieval.
        searcher (object): The **retrieval index** used for document lookup.

    Example:
        ```python
        from rankify.dataset.dataset import Document, Question
        from rankify.retrievers.hyde_retriever import HydeRetriever

        retriever = HydeRetriever(n_docs=10, num_generated_docs=2, llm_model="gpt-4")
        documents = [Document(question=Question("What are the effects of global warming?"))]

        retrieved_documents = retriever.retrieve(documents)
        print(retrieved_documents[0].contexts[0].text)
        ```
    """

    def __init__(self, model = "facebook/contriever-msmarco", n_docs: int = 100,batch_size: int = 32, device: str = "cuda", index_type: str = "wiki", **kwargs):
        """
        Initializes the **HyDE Retriever**.

        Args:
            model (str, optional): Name of the **dense retrieval model** (default: `"facebook/contriever-msmarco"`).
            n_docs (int, optional): Number of **top documents** to retrieve per query (default: `100`).
            batch_size (int, optional): Number of **queries processed per batch** (default: `32`).
            device (str, optional): Computing device (`"cuda"` or `"cpu"`, default: `"cuda"`).
            index_type (str, optional): The **type of retrieval index** (`"wiki"` or `"msmarco"`, default: `"wiki"`).
            **kwargs: Additional optional arguments:
                - `task` (str): Type of **retrieval task** (default: `"web search"`).
                - `api_key` (str, optional): API key for OpenAI-based generation.
                - `llm_model` (str, optional): LLM model for hypothetical document generation.
                - `base_url` (str, optional): Custom API base URL for OpenAI requests.
                - `num_generated_docs` (int, optional): Number of **hypothetical documents** to generate per query.
                - `max_token_generated_docs` (int, optional): Maximum **token length** for generated documents.

        Raises:
            ValueError: If any required **retrieval component** fails to initialize.
        """
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
        """
        Retrieves **contexts** for each document using **HyDE-based query expansion**.

        The process involves:
        1. **Generating hypothetical documents** using an LLM.
        2. **Encoding** both the original query and the generated documents.
        3. **Averaging the embeddings** to obtain a **refined query representation**.
        4. **Retrieving top documents** using FAISS-based search.

        Args:
            documents (List[Document]): A list of **Document objects** containing queries.

        Returns:
            List[Document]: The list of **Document objects**, each updated with retrieved **contexts**.
        
        Raises:
            ValueError: If retrieval or embedding computation fails.
        """
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

                
            





