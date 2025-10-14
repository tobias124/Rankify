import torch
from rankify.dataset.dataset import Document, Question, Answer, Context
from rankify.generator.generator import Generator
from rankify.metrics.generator_metrics import GeneratorMetrics

# ----- 1) build a doc with GOLD answers -----
question = Question("What is the capital of France?")
answers  = Answer(["Paris"])  # <— important: gold(s) used by EM/F1 etc.
contexts = [
    Context(id=1, title="France",  text="The capital of France is Paris.",   score=0.9),
    Context(id=2, title="Germany", text="Berlin is the capital of Germany.", score=0.5),
]
doc = Document(question=question, answers=answers, contexts=contexts)

# ----- 2) generate with your preferred backend -----
generator = Generator(
    method="basic-rag",
    model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
    backend="huggingface",
    torch_dtype=torch.float16
)
predictions = generator.generate([doc])   # -> ["Paris"] (ideally)

# ----- 3) evaluate generator outputs -----
gen_metrics = GeneratorMetrics([doc])
scores = gen_metrics.all(predictions)     # dict of EM/F1/etc.
print("Generator metrics:", scores)


from rankify.metrics.ragas_bridge import RagasModels

# Hugging Face judge (no API key needed if model is local)
ragas = RagasModels(
    llm_kind="hf",
    llm_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
    embeddings_kind="hf",
    embeddings_name="sentence-transformers/all-MiniLM-L6-v2",
)

scores_ragas = gen_metrics.all(predictions, ragas_models=ragas)
print("Generator metrics + RAGAS:", scores_ragas)
