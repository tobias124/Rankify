from rankify.generator.models.base_rag_model import BaseRAGModel
from rankify.generator.prompt_generator import PromptGenerator

class HuggingFaceModel(BaseRAGModel):
    def __init__(self, model_name: str, tokenizer, model, prompt_generator: PromptGenerator):
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.model = model
        self.prompt_generator = prompt_generator

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response using Hugging Face's model."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device) # ensure inputs are on the same device as the model
        # define generation parameters defaults TODO: should be excluded into config
        kwargs.setdefault("max_length", 128)
        kwargs.setdefault("temperature", 0.7)
        

        outputs = self.model.generate(**inputs, **kwargs)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)