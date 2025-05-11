from rankify.dataset.dataset import Document, Question, Answer, Context
from rankify.generator.generator import Generator

# Define question and answer
question = Question("What is the capital of France?")
answers = Answer(["Paris"])
contexts = [
    Context(id=1, title="France", text="The capital of France is Paris.", score=0.9),
    Context(id=2, title="Germany", text="Berlin is the capital of Germany.", score=0.5)
]

# Construct document
doc = Document(question=question, answers=answers, contexts=contexts)

# Initialize Generator (e.g., Meta Llama)
generator = Generator(method="basic-rag", model_name='tiiuae/falcon-rw-1b', backend="huggingface")

# Generate answer
generated_answers = generator.generator.answer_question([doc])
print(generated_answers)  # Output: ["Paris"]
