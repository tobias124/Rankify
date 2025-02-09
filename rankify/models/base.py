from abc import ABC, abstractmethod
from rankify.dataset.dataset import Document

class BaseRanking(ABC):
    """
    An abstract base class for implementing different ranking models.

    This class defines the interface for all ranking models, ensuring that all subclasses implement the required methods.

    Attributes
    ----------
    method : str
        The name of the ranking method.
    model_name : str
        The name of the model being used for ranking.
    api_key : str, optional
        An optional API key for accessing remote models or services.
    """
    @abstractmethod
    def __init__(self, method: str= None, model_name: str= None, api_key: str= None, **kwargs) ->None:
        """
        Initializes the base ranking model.

        Parameters
        ----------
        method : str, optional
            The name of the ranking method (default is None).
        model_name : str, optional
            The name of the model being used for ranking (default is None).
        api_key : str, optional
            An optional API key for accessing remote models or services (default is None).
        """
        pass

    @abstractmethod
    def rank(self, documents: list[Document] ):
        """
        Abstract method to rank a list of documents.

        Parameters
        ----------
        document : list of Document
            A list of Document instances that need to be ranked.

        Examples
        --------
        This method should be implemented by subclasses to provide a ranking mechanism.
        
        >>> class MyRanking(BaseRanking):
        >>>     def __init__(self, method, model_name):
        >>>         super().__init__(method, model_name)
        >>>     def rank(self, document):
        >>>         # Ranking implementation here
        >>>         pass
        """
        pass

