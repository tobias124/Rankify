from rankify.dataset.dataset import Document, Question, Answer, Context
from rankify.generator.generator import Generator

# Define question and answer
question = Question("Who is the Soccer world cup winner of 2014?")
#answers = Answer(["Paris"])
answers=Answer("")
contexts = [
    Context(id=1, title="France", text="The capital of France is Paris.", score=0.9),
    Context(id=2, title="Germany", text="Berlin is the capital of Germany.", score=0.5)
]

# Construct document
doc = Document(question=question, answers=answers, contexts=contexts)

# Initialize Generator (e.g., Meta Llama)
generator = Generator(method="basic-rag", model_name='meta-llama/Meta-Llama-3.1-8B-Instruct', backend="huggingface")

# Generate answer
generated_answers = generator.generator.answer_question([doc], max_new_tokens=100)
print(generated_answers)  # Output: ["Paris"]
