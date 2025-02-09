import copy
from rankify.models.base import BaseRanking
from rankify.dataset.dataset import Document
from typing import List
from rankify.utils.pre_defind_models import HF_PRE_DEFIND_MODELS, URL
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm  # Import tqdm for progress tracking


class RankGPT(BaseRanking):
    """
    Implements **RankGPT** `[19]`_, a **GPT-based re-ranking method** for passage retrieval.

    .. _[19]: https://arxiv.org/abs/2304.09542

    This method **leverages GPT models** (e.g., GPT-4, GPT-3.5) to rerank passages based on query relevance.
    It applies **sliding window** and **permutation-based ranking** strategies for more efficient re-ranking.

    References
    ----------
    .. [19] Sun, W. et al. (2023). Is ChatGPT Good at Search? Investigating Large Language Models as Re-ranking Agents.

    Attributes
    ----------
    method : str
        The ranking method used (either `'rankgpt'` or `'rankgpt-api'`).
    model_name : str
        The name of the pre-trained **GPT model**.
    api_key : str
        The API key for accessing the GPT model via API.
    model : PreTrainedModel
        The **GPT model** for ranking.
    tokenizer : PreTrainedTokenizer
        The tokenizer for encoding the input and decoding the output.
    use_gpu : bool
        Whether to use **GPU** for inference.
    use_bf16 : bool
        Whether to use **bfloat16** for faster inference.

    Examples
    --------
    Basic Usage:

    >>> from rankify.dataset.dataset import Document, Question, Context
    >>> from rankify.models.reranking import Reranking
    >>>
    >>> # Define a query and contexts
    >>> question = Question("What is the capital of France?")
    >>> contexts = [
    >>>     Context(text="Paris is the capital of France.", id=0),
    >>>     Context(text="Berlin is the capital of Germany.", id=1),
    >>>     Context(text="Madrid is the capital of Spain.", id=2),
    >>> ]
    >>> document = Document(question=question, contexts=contexts)
    >>>
    >>> # Initialize RankGPT
    >>> model = Reranking(method='rankgpt', model_name='llamav3.1-8b')
    >>> model.rank([document])
    >>>
    >>> # Print reordered contexts
    >>> print("Reordered Contexts:")
    >>> for context in document.reorder_contexts:
    >>>     print(context.text)
    """

    def __init__(self, method: str = None, model_name: str = None, api_key: str = None , **kwargs):
        """
        Initializes a RankGPT instance.

        Parameters
        ----------
        method : str
            The ranking method used (`'rankgpt'` or `'rankgpt-api'`).
        model_name : str
            The name of the **GPT model**.
        api_key : str, optional
            The API key for **GPT-based API ranking**.
        """
        self.method = method
        self.window_size = kwargs.get("window_size", 4) 
        self.step = kwargs.get("step", 1)
        self.api_key = api_key
        self.model = None
        self.tokenizer = None
        self.use_gpu = True
        self.use_bf16 = True
        if self.method == 'rankgpt-api':
            self.model_name = URL[model_name]['model_name']
            self.url = URL[model_name]['url']
        else:
            #print(model_name)
            self.model_name = model_name

        self._load(model_name)

    def _load(self, model_name: str = None) -> None:
        """
        Loads the GPT model and tokenizer.

        Parameters
        ----------
        model_name : str, optional
            The name of the model to load.
        """
        if self.method == 'rankgpt-api':
            self.model = URL[model_name]['class'](self.api_key, self.url)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.use_bf16 else torch.float32
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.use_gpu and torch.cuda.is_available():
                self.model = self.model.cuda()

    def rank(self, documents: List[Document])-> List[Document]:
        """
        Ranks the contexts within each document.

        Parameters
        ----------
        documents : list of Document
            The list of documents to rank.

        Returns
        -------
        List[Document]
            Documents with reordered contexts after ranking.
        """
        for document in tqdm(documents, desc="Reranking Documents"):
            reorder_contexts = self.sliding_windows(document, rank_start=0, rank_end=len(document.contexts), window_size=self.window_size, step=self.step)
            document.reorder_contexts = reorder_contexts
        return documents
    def sliding_windows(self, item=None, rank_start=0, rank_end=100, window_size=20, step=10):
        """
        Applies sliding window ranking.

        Parameters
        ----------
        item : Document
            The document to be ranked.
        rank_start : int
            The start index for ranking.
        rank_end : int
            The end index for ranking.
        window_size : int
            The size of the ranking window.
        step : int
            The step size for moving the window.

        Returns
        -------
        List[Context]
            The reordered contexts.
        """
        item.reorder_contexts = item.contexts
        item = copy.deepcopy(item)
        end_pos = rank_end
        start_pos = rank_end - window_size
        while start_pos >= rank_start:
            start_pos = max(start_pos, rank_start)
            item = self.permutation_pipeline(item=item, rank_start=start_pos, rank_end=end_pos)
            end_pos = end_pos - step
            start_pos = start_pos - step
        return item.reorder_contexts

    def permutation_pipeline(self, item=None, rank_start=0, rank_end=100):
        """
        Creates permutation instructions, runs LLM, and reorders the contexts accordingly.

        Parameters
        ----------
        item : Document
            The document whose contexts are to be ranked.
        rank_start : int, optional
            The start index for ranking (default is 0).
        rank_end : int, optional
            The end index for ranking (default is 100).

        Returns
        -------
        Document
            The updated document with reordered contexts.

        Examples
        --------
        >>> rank_gpt = RankGPT(method='rankgpt', model_name='gpt-3.5-turbo')
        >>> updated_document = rank_gpt.permutation_pipeline(document, rank_start=0, rank_end=10)
        """
        messages = self.create_permutation_instruction(item=item, rank_start=rank_start, rank_end=rank_end)
        
        permutation = self.run_llm(messages)
        
        item = self.receive_permutation(item=item, permutation=permutation, rank_start=rank_start, rank_end=rank_end)
    
        ###asdada
        return item

    def create_permutation_instruction(self, item=None, rank_start=0, rank_end=100):
        """
        Creates permutation instructions to pass to the LLM for ranking.

        Parameters
        ----------
        item : Document
            The document whose contexts are to be ranked.
        rank_start : int, optional
            The start index for ranking (default is 0).
        rank_end : int, optional
            The end index for ranking (default is 100).

        Returns
        -------
        list of dict
            Messages used to prompt the LLM for ranking.

        Examples
        --------
        >>> rank_gpt = RankGPT(method='rankgpt', model_name='gpt-3.5-turbo')
        >>> messages = rank_gpt.create_permutation_instruction(document, rank_start=0, rank_end=10)
        """
        query = item.question.question
        num = len(item.contexts[rank_start: rank_end])
        max_length = 300
        messages = self.get_prefix_prompt(query, num)
        rank = 0
        for hit in item.contexts[rank_start: rank_end]:
            rank += 1
            content = hit.text
            content = content.replace("Title: Content: ", "")
            content = content.strip()
            
            content = " ".join(content.split()[: int(max_length)])
            messages.append({"role": "user", "content": f"[{rank}] {content}"})
            messages.append({"role": "assistant", "content": f"Received passage [{rank}]."})
        messages.append({"role": "user", "content": self.get_post_prompt(query, num)})

        return messages

    def run_llm(self, messages):
        """
        Runs the GPT model.

        Parameters
        ----------
        messages : List[Dict[str, str]]
            The messages forming the ranking prompt.

        Returns
        -------
        str
            The ranked order response.
        """
        if self.method == 'rankgpt-api':
            response = self.model.chat(model=self.model_name, messages=messages, temperature=0, return_text=True)
        else:
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(self.model.device)
            terminators = [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
            with torch.no_grad():
                responses = self.model.generate(
                    input_ids,
                    max_new_tokens=128,
                    eos_token_id=terminators,
                    do_sample=False,
                    temperature=1.0,   # Default value for deterministic generation
                    top_p=1.0          # Default value, no effect when do_sample=False
                )
            response = responses[0][input_ids.shape[-1]:]
            response = self.tokenizer.decode(response, skip_special_tokens=True)
        return response

    def receive_permutation(self, item, permutation, rank_start=0, rank_end=100):
        """
        Processes the ranking output from the LLM to reorder contexts within a document.

        Parameters
        ----------
        item : Document
            The document containing contexts to be reordered.
        permutation : str
            The output from the LLM, representing the new ranking order.
        rank_start : int, optional
            The start index for the range of contexts to be ranked (default is 0).
        rank_end : int, optional
            The end index for the range of contexts to be ranked (default is 100).

        Returns
        -------
        Document
            The updated document with reordered contexts based on the ranking output.

        Examples
        --------
        >>> rank_gpt = RankGPT(method='rankgpt', model_name='gpt-3.5-turbo')
        >>> permutation = "[2] > [1]"
        >>> updated_document = rank_gpt.receive_permutation(document, permutation, rank_start=0, rank_end=10)
        """
        response = self.clean_response(permutation)
        response = [int(x) - 1 for x in response.split()]
        response = self.remove_duplicate(response)
        cut_range = copy.deepcopy(item.contexts[rank_start:rank_end])
        #print(cut_range)
        original_rank = [tt for tt in range(len(cut_range))]
        response = [ss for ss in response if ss in original_rank]
        response = response + [tt for tt in original_rank if tt not in response]
        for j, x in enumerate(response):
            #print(cut_range[x], x)
            item.reorder_contexts[j + rank_start] = copy.deepcopy(cut_range[x])
            if hasattr(item.reorder_contexts[j + rank_start], 'rank'):
                item.reorder_contexts[j + rank_start].rank = cut_range[j].rank

            if hasattr(item.reorder_contexts[j + rank_start], 'score'):
                item.reorder_contexts[j + rank_start].score = cut_range[j].score
        return item

    def get_prefix_prompt(self, query, num):
        """
        Generates the prefix for prompting the LLM.

        Parameters
        ----------
        query : str
            The query to rank contexts against.
        num : int
            The number of passages to rank.

        Returns
        -------
        List[Dict[str, str]]
            The initial ranking prompt.
        """
        return [{'role': 'system',
                 'content': "You are RankGPT, an intelligent assistant that can rank passages based on their relevancy to the query."},
                {'role': 'user',
                 'content': f"I will provide you with {num} passages, each indicated by number identifier []. \nRank the passages based on their relevance to query: {query}."},
                {'role': 'assistant', 'content': 'Okay, please provide the passages.'}]

    def clean_response(self, response: str):
        """
        Cleans the ranking response.

        Parameters
        ----------
        response : str
            The raw ranking response.

        Returns
        -------
        str
            The cleaned response with ranking order.
        """
        new_response = ''
        for c in response:
            if not c.isdigit():
                new_response += ' '
            else:
                new_response += c
        new_response = new_response.strip()
        return new_response

    def remove_duplicate(self, response):
        """
        Removes duplicates from the ranking output to ensure all contexts are ranked only once.

        Parameters
        ----------
        response : list of int
            The list of context indices as generated by the LLM.

        Returns
        -------
        list of int
            The list of unique context indices.

        Examples
        --------
        >>> rank_gpt = RankGPT(method='rankgpt', model_name='gpt-3.5-turbo')
        >>> response = [2, 2, 1, 1]
        >>> unique_response = rank_gpt.remove_duplicate(response)
        >>> print(unique_response)
        [2, 1]
        """
        new_response = []
        for c in response:
            if c not in new_response:
                new_response.append(c)
        return new_response
    def get_post_prompt(self, query, num):
        """
        Generates the final prompt for the LLM to request ranking output.

        Parameters
        ----------
        query : str
            The query to rank contexts against.
        num : int
            The number of passages to be ranked.

        Returns
        -------
        str
            The prompt to finalize the ranking request.

        Examples
        --------
        >>> rank_gpt = RankGPT(method='rankgpt', model_name='gpt-3.5-turbo')
        >>> post_prompt = rank_gpt.get_post_prompt("What is the capital of France?", 2)
        >>> print(post_prompt)
        'Search Query: What is the capital of France...'
        """
        return f"Search Query: {query}. \nRank the {num} passages above based on their relevance to the search query. The passages should be listed in descending order using identifiers. The most relevant passages should be listed first. The output format should be [] > [], e.g., [1] > [2]. Only response the ranking results, do not say any word or explain."
