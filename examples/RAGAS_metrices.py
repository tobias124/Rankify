import torch
from rankify.dataset.dataset import Document, Question, Answer, Context
from rankify.generator.generator import Generator
from rankify.metrics.generator_metrics import GeneratorMetrics
from rankify.metrics.ragas_bridge import RagasModels

# Build document
question = Question("What is the capital of France?")
answers = Answer(["Paris"])
contexts = [
    Context(id=1, title="France", text="The capital of France is Paris.", score=0.9),
    Context(id=2, title="Germany", text="Berlin is the capital of Germany.", score=0.5),
]
doc = Document(question=question, answers=answers, contexts=contexts)

# Generate predictions
generator = Generator(
    method="basic-rag",
    model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
    backend="huggingface",
    torch_dtype=torch.float16
)
predictions = generator.generate([doc])
print(f"Generated: {predictions}")

# Evaluate with HuggingFace - PROPER TIMEOUT SETTINGS
gen_metrics = GeneratorMetrics([doc])

# Configure RAGAS with longer timeouts for HuggingFace
ragas_hf = RagasModels(
    llm_kind="hf",
    llm_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
    embeddings_kind="hf",
    embeddings_name="sentence-transformers/all-MiniLM-L6-v2",
    torch_dtype="float16",
    max_new_tokens=256,  # Shorter outputs = faster
    timeout=180,  # 3 minutes per metric call
    max_retries=1,  # Only retry once
    max_workers=2,  # Fewer parallel workers for stability
)

# Option 1: Use default fast metrics (faithfulness + context_precision only)
print("\n=== RAGAS with Default Fast Metrics ===")
scores_fast = gen_metrics.all(predictions, ragas_models=ragas_hf)
for k, v in scores_fast.items():
        print(f"{k}: {v:.4f}")

# Option 2: Specify exactly which metrics you want
print("\n=== RAGAS with Specific Metrics ===")
scores_specific = gen_metrics.ragas_generator(
    predictions,
    judge=ragas_hf,
    metrics=["faithfulness", "response_relevancy", "context_precision", "context_recall"]  # Just one metric for testing
)
for k, v in scores_specific.items():
    print(f"{k}: {v:.4f}")

# Option 3: Use OpenAI for faster evaluation (if you have API key)
print("\n=== RAGAS with OpenAI (Much Faster) ===")
ragas_openai = RagasModels(
    llm_kind="openai",
    llm_name="gpt-4o-mini",
    timeout=30,  # Much shorter timeout for API calls
)
scores_openai = gen_metrics.all(predictions, ragas_models=ragas_openai)
for k, v in scores_openai.items():
    if k.startswith("ragas_"):
        print(f"{k}: {v:.4f}")