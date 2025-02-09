from rankify.dataset.dataset import Document

from rankify.utils.pre_defind_models import HF_PRE_DEFIND_MODELS
from rankify.utils.pre_defined_methods import METHOD_MAP

class Reranking:
    """
    A class for reranking documents using different ranking models.

    Attributes
    ----------
    method : str
        The name of the reranking method to be used.
    model : str
        The name of the model used for reranking.
    api_key : str, optional
        An optional API key for accessing remote models or services.
    ranker : BaseRanking
        An instance of a ranking model that will be used to perform the reranking.
    """
    def __init__(self, method: str= None, model_name: str= None, api_key: str = None , **kwargs):
        """
        Initializes a Reranking instance.

        Parameters
        ----------
        method : str, optional
            The name of the reranking method to be used (default is None).
        model_name : str, optional
            The name of the model to be used for reranking (default is None).
        api_key : str, optional
            An optional API key for accessing remote models or services (default is None).

        Examples
        --------
        >>> model = Reranking(method='listt5', model_name='listt5-base')
        """
        self.method = method
        self.model = model_name
        self.api_key= api_key
        self.kwargs = kwargs
        self.ranker = self.initialize()
    
    def initialize(self):
        """
        Initializes the ranking model based on the specified method and model name.

        Returns
        -------
        BaseRanking
            An instance of a ranking model.

        Raises
        ------
        ValueError
            If the specified ranking method is not supported.

        Examples
        --------
        >>> model = Reranking(method='listt5', model_name='listt5-base')
        >>> ranker = model.initialize()
        """
        model_name= HF_PRE_DEFIND_MODELS[self.method][self.model]
        #print(model_name)
        #aaaaa
        if self.method in METHOD_MAP:
            return METHOD_MAP[self.method](method=self.method,model_name=model_name, api_key=self.api_key, **self.kwargs)
        else:
            raise ValueError(f"Ranking method {self.method} is not supported or installed. please install the full version of Rankify")

    def rank(self, documents: list[Document]):
        """
        Reranks a list of documents using the specified ranking model.

        Parameters
        ----------
        documents : list of Document
            A list of Document instances that need to be reranked.

        Returns
        -------
        list of Document
            The reranked list of Document instances.

        Examples
        --------
        >>> question = Question("When did Thomas Edison invent the light bulb?")
        >>> answers = Answer(["1879"])
        >>> contexts = [
        ...     Context(text="Lightning strike at Seoul National University", id=1),
        ...     Context(text="Thomas Edison invented the light bulb in 1879", id=5)
        ... ]
        >>> document = Document(question=question, answers=answers, contexts=contexts)
        >>> model = Reranking(method='listt5', model_name='listt5-base')
        >>> model.rank([document])
        """
        return self.ranker.rank(documents=documents)