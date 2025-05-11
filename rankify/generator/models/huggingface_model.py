from typing import List
from rankify.generator.base_rag_model import BaseRAGModel
from rankify.generator.prompt_generator import PromptGenerator

class HuggingFaceModel(BaseRAGModel):
    def __init__(self, model_name: str, tokenizer, model, prompt_generator: PromptGenerator):
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.model = model
        self.prompt_generator = prompt_generator

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response using Hugging Face's model."""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, **kwargs)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def embed(self, text: str, **kwargs) -> List[float]:
        """Generate embeddings using Hugging Face's model."""
        # Example implementation (replace with actual embedding logic)
        return [0.4, 0.5, 0.6]  # Dummy embedding