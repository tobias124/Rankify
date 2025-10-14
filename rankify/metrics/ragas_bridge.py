# rankify/metrics/ragas_bridge.py
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

def _get_doc_question(doc):
    # try common attribute names
    for attr in ["question", "query", "user_input"]:
        if hasattr(doc, attr):
            return getattr(doc, attr)
    raise AttributeError("Document missing .question/.query")

def _ctx_text(ctx):
    # support different context object shapes
    for attr in ["text", "content", "passage", "chunk", "raw"]:
        if hasattr(ctx, attr):
            return getattr(ctx, attr)
    # fall back to str
    return str(ctx)

def _gold_answers(doc):
    # you already use this in BaseMetric.get_dataset_answer
    ans = getattr(doc, "answers", None)
    if hasattr(ans, "answers"):
        return list(ans.answers)
    if isinstance(ans, (list, tuple)):
        return list(ans)
    if isinstance(ans, str):
        return [ans]
    return [str(ans)]

def _contexts(doc, use_reordered: bool):
    arr = None
    if use_reordered and getattr(doc, "reorder_contexts", None):
        arr = doc.reorder_contexts
    else:
        arr = getattr(doc, "contexts", None)
    if not arr:
        return []
    return [_ctx_text(c) for c in arr]

@dataclass
class RagasModels:
    """Lazy optional construction of judge LLM and embeddings for RAGAS."""
    provider: str = "langchain"           # "langchain" is most flexible
    llm_kind: str = "openai"              # "openai" or "hf"
    llm_name: Optional[str] = None        # e.g., "gpt-4o-mini" or "meta-llama/Meta-Llama-3.1-8B-Instruct"
    llm_api_key: Optional[str] = None     # env var by default
    embeddings_kind: str = "hf"           # "openai" or "hf"
    embeddings_name: Optional[str] = "sentence-transformers/all-MiniLM-L6-v2"

    def build(self):
        """
        Returns (ragas_llm, ragas_embeddings) objects compatible with ragas.evaluate.
        Uses RagAS' LangChain wrappers so you can choose OpenAI or HF locally.
        """
        # import here so your code doesn't hard-depend if user disables RAGAS
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper

        # --- LLM ---
        if self.llm_kind == "openai":
            # pip install langchain-openai openai
            from langchain_openai import ChatOpenAI
            judge = ChatOpenAI(
                model=self.llm_name or "gpt-4o-mini",
                temperature=0.0,
                api_key=self.llm_api_key,  # or None to use env
            )
        else:
            # Hugging Face Transformers pipeline via LangChain
            # pip install langchain-huggingface transformers accelerate
            from langchain_huggingface import ChatHuggingFace
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            model_name = self.llm_name or "meta-llama/Meta-Llama-3.1-8B-Instruct"
            tok = AutoTokenizer.from_pretrained(model_name)
            mdl = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
            pipe = pipeline("text-generation", model=mdl, tokenizer=tok, max_new_tokens=512)
            judge = ChatHuggingFace(pipeline=pipe)

        ragas_llm = LangchainLLMWrapper(judge)

        # --- Embeddings ---
        if self.embeddings_kind == "openai":
            # pip install langchain-openai
            from langchain_openai import OpenAIEmbeddings
            emb = OpenAIEmbeddings(model=self.embeddings_name or "text-embedding-3-large", api_key=self.llm_api_key)
        else:
            # pip install langchain-huggingface sentence-transformers
            from langchain_huggingface import HuggingFaceEmbeddings
            emb = HuggingFaceEmbeddings(model_name=self.embeddings_name)

        ragas_embeddings = LangchainEmbeddingsWrapper(emb)
        return ragas_llm, ragas_embeddings


def build_ragas_dataset(documents, predictions: List[str], use_reordered: bool):
    """
    Build a RagasDataset (SingleTurnSample list) from rankify docs + predictions.
    """
    from ragas import RagasDataset, SingleTurnSample
    samples = []
    for doc, pred in zip(documents, predictions):
        samples.append(
            SingleTurnSample(
                user_input=_get_doc_question(doc),
                response=pred or "",
                retrieved_contexts=_contexts(doc, use_reordered=use_reordered),
                reference=" ".join(_gold_answers(doc))  # Ragas supports 'reference' on SingleTurnSample
            )
        )
    return RagasDataset.from_list(samples)
