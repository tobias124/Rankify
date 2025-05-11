

from rankify.generator.base_rag_model import BaseRAGModel
from rankify.generator.models.huggingface_model import HuggingFaceModel
from rankify.generator.prompt_generator import PromptGenerator


def model_factory(model_name: str, backend: str, method: str, **kwargs) -> BaseRAGModel:
    prompt_generator = PromptGenerator(model_type=model_name, method=method)
    if backend == "openai":
        return OpenAIModel(model_name, kwargs["api_key"], prompt_generator)
    elif backend == "huggingface":
        return HuggingFaceModel(model_name, kwargs["tokenizer"], kwargs["model"], prompt_generator)
    #elif backend == "vllm":
    #    return VLLMModel(model_name, kwargs["engine"], prompt_generator)
    else:
        raise ValueError(f"Unsupported backend: {backend}")