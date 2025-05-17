import torch
from rankify.generator.models.fid_model import FiDModel
from rankify.generator.models.openai_model import OpenAIModel
from rankify.generator.models.base_rag_model import BaseRAGModel
from rankify.generator.models.huggingface_model import HuggingFaceModel
from rankify.generator.prompt_generator import PromptGenerator

from transformers import AutoTokenizer, AutoModelForCausalLM

from rankify.utils.generator.huggingface_models.model_utils import load_model, load_tokenizer

def model_factory(model_name: str, backend: str, method: str, use_litellm=False, **kwargs) -> BaseRAGModel:
    prompt_generator = PromptGenerator(model_type=model_name, method=method)
    if backend == "openai":
        return OpenAIModel(model_name, kwargs["api_keys"], prompt_generator, use_litellm=use_litellm)
    elif backend == "huggingface":
        tokenizer = load_tokenizer(model_name)
        model = load_model(model_name, **kwargs) 
        return HuggingFaceModel(model_name, tokenizer, model, prompt_generator)
    elif backend == "fid":
        return FiDModel(method, model_name, **kwargs)
    else:
        raise ValueError(f"Unsupported backend: {backend}")