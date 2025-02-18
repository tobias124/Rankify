from rankify.utils.dataset.utils import get_datasets_info
from rankify.utils.dataset.download import DownloadManger
import json
from typing import List,Optional,Dict
import os, requests
from tqdm import tqdm
import csv

class Question:
    """
    Represents a question with automatic validation.

    Attributes:
        question (str): The text of the question.
    """
    def __init__(self, question: str ) -> None:
        """
        Initializes a Question instance.

        Args:
            question (str): The text of the question.

        Example:
            ```python
            q = Question("What is the capital of France?")
            print(q)  # Output: Question: What is the capital of France?
            ```
        """
        self.question = self.check_question(question)
    @classmethod
    def check_question(cls,question) -> str:
        """
        Ensures the question ends with a question mark.

        Args:
            question (str): The text of the question.

        Returns:
            str: The question with a question mark at the end if it was missing.

        Example:
            ```python
            Question.check_question("What is the capital of France")
            # Output: 'What is the capital of France?'
            ```
        """
        cls.question = question if question.endswith("?") else question +"?"
        return cls.question
    
    def __str__(self) -> str:
        """
        Returns a string representation of the Question instance.

        Returns:
            str: The formatted question.

        Example:
            ```python
            q = Question("What is the capital of France?")
            str(q)  # Output: 'Question: What is the capital of France?'
            ```
        """
        return f"Question: {self.question}"
    
class Answer:
    """
    Represents answers to a question.

    Attributes:
        answers (list[str]): A list of possible answers.
    """
    def __init__(self, answers:list=None) -> None:
        """
        Initializes an Answer instance.

        Args:
            answers (list[str] or str, optional): A list of possible answers. Defaults to None.

        Example:
            ```python
            a = Answer(["Paris", "Lyon"])
            print(a)  
            # Output:
            # Answer:
            # - Paris
            # - Lyon
            ```
        """
        if isinstance(answers, str):  # If it's a string, convert it to a list
            self.answers = [answers]
        if isinstance(answers, int):
             self.answers = [str(answers)]
        elif isinstance(answers, list):  # If it's a list, ensure all elements are strings
            self.answers = [str(answer) for answer in answers]
        else:  # If it's neither, initialize with an empty list
            self.answers = []

    def __str__(self) -> str:
        """
        Returns a string representation of the Answer instance.

        Returns:
            str: The formatted answers.

        Example:
            ```python
            a = Answer(["Paris", "Lyon"])
            str(a)  
            # Output: 
            # Answer: 
            # - Paris
            # - Lyon
            ```
        """
        return f"Answer: \n- "+ "\n- ".join(self.answers)

class Context:
    """
    Represents a context with metadata such as score and title.

    Attributes:
        score (float, optional): The relevance score of the context.
        has_answer (bool, optional): Whether the context contains an answer.
        id (int, optional): The identifier of the context.
        title (str, optional): The title of the context.
        text (str, optional): The text of the context.
    """
    def __init__(self, score: float=None, has_answer: bool=None, id: int=None, title: str=None, text: str=None)-> None:
        """
        Initializes a Context instance.

        Args:
            score (float, optional): The relevance score.
            has_answer (bool, optional): Whether the context contains an answer.
            id (int, optional): The identifier of the context.
            title (str, optional): The title of the context.
            text (str, optional): The text of the context.

        Example:
            ```python
            c = Context(score=0.9, has_answer=True, id=1, title="Paris", text="The capital of France is Paris.")
            print(c)
            ```
        """
        self.score: Optional[float] = score
        self.has_answer: Optional[bool] = has_answer
        self.id: Optional[int] = id
        self.title: Optional[str] = title
        self.text: Optional[str] = text

    def to_dict(self, save_text: bool=False) -> Dict[str, Optional[object]]:

        """
        Converts the Context instance to a dictionary.

        Args:
            save_text (bool): Whether to include text in the output dictionary.

        Returns:
            dict: The context data.

        Example:
            ```python
            c = Context(score=0.9, has_answer=True, id=1, title="Paris", text="The capital of France is Paris.")
            print(c.to_dict())
            ```
        """
        context_dict = {
            "score": float(self.score) if self.score is not None else None,
            "has_answer": self.has_answer,
            "id": self.id,
            }

        # Include 'text' only if save_text is True
        if save_text:
            context_dict["text"] = self.text
            context_dict["title"] =  self.title

        return context_dict
    def __str__(self) -> str:
        """
        Returns a string representation of the Context instance.

        Returns:
            str: The formatted context.

        Example:
            ```python
            c = Context(score=0.9, has_answer=True, id=1, title="Paris", text="The capital of France is Paris.")
            print(str(c))
            ```
        """
        return f"ID: {self.id}\nHas Answer: {self.has_answer}\nTitle: {self.title}\nText: {self.text}\nScore: {self.score}"

class Document:
    """
    A class to represent a document, consisting of a question, answers, and contexts.

    Attributes
    ----------
    question : Question
        The question associated with the document.
    answers : Answer
        The answers to the question.
    contexts : list of Context
        A list of contexts related to the question.
    reorder_contexts : list of Context, optional
        A reordered list of contexts based on their relevance (default is None).
    """
    def __init__(self, question: Question, answers: Answer, contexts: list = None) -> None:
        """
        Initializes a Document instance.

        Parameters
        ----------
        question : Question
            The question associated with the document.
        answers : Answer
            The answers to the question.
        contexts : list of Context
            A list of contexts related to the question.

        Examples
        --------
        >>> q = Question("What is the capital of France?")
        >>> a = Answer(["Paris"])
        >>> c1 = Context(score=0.9, has_answer=True, id=1, title="Paris", text="The capital of France is Paris.")
        >>> c2 = Context(score=0.5, has_answer=False, id=2, title="Berlin", text="Berlin is the capital of Germany.")
        >>> d = Document(question=q, answers=a, contexts=[c1, c2])
        >>> print(d)
        Question: What is the capital of France?
        Answer: Paris
        Context: 
        
        ID: 1
        Has Answer: True
        Title: Paris
        Text: The capital of France is Paris.
        Score: 0.9
        
        ID: 2
        Has Answer: False
        Title: Berlin
        Text: Berlin is the capital of Germany.
        Score: 0.5
        Reorder contexts: 
        
        """        
        self.question: Question = question
        self.answers: Answer = answers
        self.contexts: List[Context] = contexts
        self.reorder_contexts: List[Context] = None

    @classmethod
    def from_dict(cls, data: dict,n_docs:int=100) -> 'Document':
        """
        Creates a Document instance from a dictionary representation.

        Parameters
        ----------
        data : dict
            A dictionary containing the question, answers, and contexts.
        n_docs : int, optional
            The number of contexts to include (default is 100).

        Returns
        -------
        Document
            A new Document instance created from the provided dictionary.

        Examples
        --------
        >>> data = {
        ...     "question": "What is the capital of France?",
        ...     "answers": ["Paris"],
        ...     "ctxs": [
        ...         {"score": 0.9, "has_answer": True, "id": 1, "title": "Paris", "text": "The capital of France is Paris."},
        ...         {"score": 0.5, "has_answer": False, "id": 2, "title": "Berlin", "text": "Berlin is the capital of Germany."}
        ...     ]
        ... }
        >>> d = Document.from_dict(data)
        >>> print(d.question)
        Question: What is the capital of France?
        """
        question = Question(data["question"])
        answers = Answer(data["answers"])
        
        
        contexts = [Context(**ctx) for ctx in data["ctxs"][:n_docs]]
        return cls(question, answers, contexts)

    def to_dict(self) -> Dict[str, Optional[object]]:
        # Convert the document into a dictionary representation
        return {
            "question": self.question.question,
            "answers": self.answers.answers,
            "contexts": [ctx.to_dict() for ctx in self.contexts]
        }
    def to_dict_reoreder(self) -> Dict[str,Optional[object]]:
        return {
            "question" : self.question.question,
            "answers" : self.answers.answers,
            "contexts" : [ctx.to_dict() for ctx in self.reorder_contexts]
        }
    def __str__(self) -> str:
        contexts_str = "\n\n".join([str(ctx) for ctx in self.contexts])
        reorder_contexts_str= ''
        if self.reorder_contexts is not None:
            reorder_contexts_str = "\n\n".join([str(ctx) for ctx in self.reorder_contexts])
        return f"{self.question}\n\n{self.answers}\n\nContext: \n\n{contexts_str}\nReorder contexts: \n\n{reorder_contexts_str}"

class Dataset:
    PASSAGES_URL = "https://huggingface.co/datasets/abdoelsayed/reranking-datasets/resolve/main/psgs_w100/psgs_w100.tsv?download=true"
    CACHE_DIR = os.environ.get("RERANKING_CACHE_DIR", "./cache")
    PASSAGES_FILE = os.path.join(CACHE_DIR, "psgs_w100.tsv")
    def __init__(self, retriever: str, dataset_name:str, n_docs:int = 1000) -> None:
        """
        Initializes a Dataset instance.

        Parameters
        ----------
        retriever : str
            The name of the retriever used to obtain the dataset.
        dataset_name : str
            The name of the dataset.
        n_docs : int, optional
            The number of documents to include (default is 1000).

        Examples
        --------
        >>> dataset = Dataset(retriever='bm25', dataset_name='example_dataset', n_docs=500)
        >>> print(dataset.dataset_name)
        'example_dataset'
        """
        self.dataset_name: str  = dataset_name
        self.retriever: str  = retriever
        self.n_docs: int = n_docs
        self.documents: Optional[List[Document]]  = None
        self.id_to_text_title: Optional[Dict] = None

    def _ensure_passages_downloaded(self) -> None:
        """
        Ensures that the passages TSV file is downloaded and cached.
        """
        if not os.path.exists(self.PASSAGES_FILE):
            print(f"Downloading passages from {self.PASSAGES_URL}...")
            os.makedirs(os.path.dirname(self.PASSAGES_FILE), exist_ok=True)
            response = requests.get(self.PASSAGES_URL, stream=True)
            with open(self.PASSAGES_FILE, "wb") as f:
                for chunk in tqdm(response.iter_content(chunk_size=1024), desc="Downloading Passages"):
                    f.write(chunk)
            print(f"Passages file downloaded and saved to {self.PASSAGES_FILE}")
        else:
            print(f"Passages file already exists at {self.PASSAGES_FILE}")
            
    def _load_passages_mapping(self) -> Dict:
        """
        Loads the passages TSV file and creates a mapping of IDs to text and title.
        
        Returns
        -------
        Dict
            A dictionary mapping passage IDs to their text and title.
        """
        if self.id_to_text_title is None:
            self.id_to_text_title = {}
            with open(self.PASSAGES_FILE, "r", encoding="utf-8") as tsv_file:
                reader = csv.DictReader(tsv_file, delimiter="\t")
                for row in reader:
                    ctx_id = int(row["id"])
                    self.id_to_text_title[ctx_id] = {
                        "text": row["text"],
                        "title": row["title"]
                    }
        return self.id_to_text_title
    def update_contexts_from_passages(self) -> None:
        """
        Updates the text and title fields of contexts in all documents using the passages TSV file.
        Downloads the TSV file if it's not already cached.
        """
        if not self.documents:
            raise ValueError("Dataset has not been loaded. Call `download()` to load the dataset.")

        # Ensure TSV file is downloaded
        self._ensure_passages_downloaded()
        
        # Load the mapping of IDs to text and title
        id_to_text_title = self._load_passages_mapping()
        
        # Update contexts in all documents
        for document in tqdm(self.documents, desc="Retreive Text and Title for documents", unit="doc"):
            for context in document.contexts:
                if context.id in id_to_text_title:
                    context_data = id_to_text_title[context.id]
                    context.text = context_data["text"]
                    context.title = context_data["title"]
                    
            
    def download(self, force_download:bool =True)-> List[Document]:
        """
        Downloads the dataset and loads it into memory.

        Parameters
        ----------
        force_download : bool, optional
            Whether to force downloading the dataset even if it already exists locally (default is True).

        Returns
        -------
        list of Document
            A list of Document instances loaded from the dataset.

        Examples
        --------
        >>> dataset = Dataset(retriever='bm25', dataset_name='example_dataset', n_docs=500)
        >>> documents = dataset.download()
        >>> len(documents)
        500
        """
        filepath= DownloadManger.download(self.retriever,self.dataset_name, force_download =force_download)
        self.documents= self.load_dataset(filepath, self.n_docs)
        self.update_contexts_from_passages()
        return self.documents
    @classmethod
    def load_dataset(cls, filepath:str, n_docs: int= 100) -> List[Document]:
        """
        Loads the dataset from a JSON file.

        Parameters
        ----------
        filepath : str
            The path to the JSON file containing the dataset.
        n_docs : int, optional
            The number of documents to load (default is 100).

        Returns
        -------
        list of Document
            A list of Document instances loaded from the JSON file.

        Examples
        --------
        >>> filepath = 'example_dataset.json'
        >>> documents = Dataset.load_dataset(filepath, n_docs=50)
        >>> len(documents)
        50
        """
        with open(filepath , encoding='utf-8') as file:
            data = json.load(file)
        data = [Document.from_dict(d,n_docs) for d in data]

        """print(data[0].contexts[0])
        asdsada"""
        return data
    @classmethod
    def load_dataset_qa(cls, filepath: str) -> List[Document]:
        """
        Loads a QA dataset from a JSON or JSONL file. The dataset contains only questions, optionally answers, and optionally IDs.

        Parameters
        ----------
        filepath : str
            The path to the JSON or JSONL file containing the dataset.

        Returns
        -------
        List[Document]
            A list of Document objects, each containing:
            - question (Question object)
            - answers (Answer object, if available)
            - id (if available)
            - contexts (empty list, since this is a QA-only file)
        
        Raises
        ------
        ValueError
            If the file format is not supported (only .json and .jsonl are allowed) or if the required fields are missing.
        
        Examples
        --------
        >>> documents = Dataset.load_dataset_qa('path/to/qa_dataset.jsonl')
        >>> print(documents[0])
        Document(question=Question("What is the capital of France?"), answers=Answer(["Paris"]), contexts=[])
        """
        documents = []

        # Load the file based on its extension (JSON or JSONL)
        if filepath.endswith(".json"):
            with open(filepath, encoding="utf-8") as file:
                data = json.load(file)
        elif filepath.endswith(".jsonl"):
            with open(filepath, encoding="utf-8") as file:
                data = [json.loads(line) for line in file]
        else:
            raise ValueError("Unsupported file format. Please use a JSON or JSONL file.")

        for entry in data:
            # Extract required fields
            #print(entry)
            question_text = entry.get("question")
            if not question_text:
                continue
                #raise ValueError("Each entry must have a 'question' field.")
            
            # Extract optional fields (answers and ID)

            answers = entry.get("answers", entry.get("answer", entry.get("golden_answers", [])))
            #print(answers)
            #asdadada
            if isinstance(answers, str):  
                answers = [answers]
            question_id = entry.get("id", None)

            # Create Question, Answer, and Document objects
            question_obj = Question(question_text)
            answer_obj = Answer(answers)
            document = Document(question=question_obj, answers=answer_obj, contexts=[])
            
            # Optionally store the ID in the document
            if question_id is not None:
                document.id = question_id

            documents.append(document)

        return documents
   
    def save_dataset(self,  output_path: str , save_reranked: bool= False, save_text:bool = False) -> None:
        """
        Saves the re-ranked documents in DPR format to a JSON or JSONL file.

        Parameters
        ----------
        documents : List[Document]
            A list of Document objects containing the questions, answers, and reordered contexts.
        output_path : str
            The path to the output file (must be .json or .jsonl).

        Returns
        -------
        None
        """
        dpr_data = []
        for doc in self.documents:
            dpr_entry = {
                "question": doc.question.question,
                "answers": doc.answers.answers,
                "ctxs": [ctx.to_dict(save_text) for ctx in doc.contexts]
            }
            if save_reranked:
                dpr_entry["reranked_ctxs"] = [ctx.to_dict(save_text) for ctx in doc.reorder_contexts]

            dpr_data.append(dpr_entry)

        if output_path.endswith(".json"):
            with open(output_path, 'w', encoding='utf-8') as file:
                json.dump(dpr_data, file, indent=4)
        elif output_path.endswith(".jsonl"):
            with open(output_path, 'w', encoding='utf-8') as file:
                for entry in dpr_data:
                    file.write(json.dumps(entry) + "\n")
        else:
            raise ValueError("Unsupported file format. Please use a .json or .jsonl file.")
    
    @staticmethod
    def save_documents(documents: List[Document], output_path: str, save_reranked: bool = False, save_text: bool = False) -> None:
        """
        Saves the list of Document objects in DPR format to a JSON or JSONL file.

        Parameters
        ----------
        documents : List[Document]
            A list of Document objects containing questions, answers, and contexts.
        output_path : str
            The path to save the DPR-formatted output (must be .json or .jsonl).
        save_reranked : bool, optional
            Whether to save re-ranked contexts (default is False).
        save_text : bool, optional
            Whether to save the full text of the contexts (default is False).

        Returns
        -------
        None
        """
        dpr_data = []
        for doc in documents:
            dpr_entry = {
                "question": doc.question.question,
                "answers": doc.answers.answers,
                "ctxs": [ctx.to_dict(save_text) for ctx in doc.contexts]
            }
            if save_reranked and doc.reorder_contexts is not None:
                dpr_entry["reranked_ctxs"] = [ctx.to_dict(save_text) for ctx in doc.reorder_contexts]

            dpr_data.append(dpr_entry)

        if output_path.endswith(".json"):
            with open(output_path, "w", encoding="utf-8") as file:
                json.dump(dpr_data, file, indent=4)
        elif output_path.endswith(".jsonl"):
            with open(output_path, "w", encoding="utf-8") as file:
                for entry in dpr_data:
                    file.write(json.dumps(entry) + "\n")
        else:
            raise ValueError("Unsupported file format. Please use a .json or .jsonl file.")

        print(f"Saved {len(documents)} documents to {output_path}.")
    def __len__(self) -> int:
        """
        Returns the number of documents in the dataset.

        Returns
        -------
        int
            The number of documents in the dataset.

        Examples
        --------
        >>> dataset = Dataset(retriever='bm25', dataset_name='example_dataset', n_docs=500)
        >>> dataset.download()
        >>> len(dataset)
        500
        """
        return len(self.documents)
    
    def __getitem__(self,idx) -> Document:
        """
        Returns the document at the specified index.

        Parameters
        ----------
        idx : int
            The index of the document to retrieve.

        Returns
        -------
        Document
            The Document instance at the specified index.

        Raises
        ------
        ValueError
            If the dataset has not been loaded.

        Examples
        --------
        >>> dataset = Dataset(retriever='bm25', dataset_name='example_dataset', n_docs=500)
        >>> dataset.download()
        >>> document = dataset[0]
        >>> print(document.question)
        Question: What is the capital of France?
        """
        if not self.documents:
            raise ValueError("Dataset has not been loaded. Call `download()` to load the dataset.")
        return self.documents[idx]

    
    @staticmethod
    def avaiable_dataset() -> None:
        """
        Prints information about available datasets.

        Examples
        --------
        >>> Dataset.avaiable_dataset()
        """
        get_datasets_info()



