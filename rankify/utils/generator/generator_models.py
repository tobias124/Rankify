from rankify.generator.rag_methods.basic_rag import BasicRAG
from rankify.generator.rag_methods.chain_of_thought_rag import ChainOfThoughtRAG
from rankify.generator.rag_methods.fid_rag_method import FiDRAGMethod
from rankify.generator.rag_methods.in_context_ralm_rag import InContextRALMRAG


RAG_METHODS = {
    "in-context-ralm": InContextRALMRAG,
    "fid": FiDRAGMethod,
    "basic-rag": BasicRAG,
    "chain-of-thought-rag": ChainOfThoughtRAG,
}