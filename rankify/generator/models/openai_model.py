from rankify.generator.models.base_rag_model import BaseRAGModel
from rankify.generator.prompt_generator import PromptGenerator

class OpenAIModel(BaseRAGModel):
    def __init__(self, model_name: str, api_key: str, prompt_generator: PromptGenerator):
        self.model_name = model_name
        self.api_key = api_key
        self.prompt_generator = prompt_generator

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response using OpenAI's API."""
        # Example implementation (replace with actual API call)
        return f"Generated response for prompt: {prompt}"
