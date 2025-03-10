from rankify.dataset.dataset import Document, Question, Answer, Context
from rankify.models.reranking import Reranking
from rankify.utils.pre_defind_models import HF_PRE_DEFIND_MODELS

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Sample document setup
question = Question("When did Thomas Edison invent the light bulb?")
answers = Answer(["1879"])
contexts = [
    Context(text="Lightning strike at Seoul National University", id=1),
    Context(text="Thomas Edison tried to invent a device for cars but failed", id=2),
    Context(text="Coffee is good for diet", id=3),
    Context(text="Thomas Edison invented the light bulb in 1879", id=4),
    Context(text="Thomas Edison worked with electricity", id=5),
]
document = Document(question=question, answers=answers, contexts=contexts)

# Function to test a reranking model
def test_reranking_model(model_category, model_name):
    reranker = None 
    try:
        print(f"Testing {model_category}: {model_name} ...")
        if model_category =="apiranker" or  model_category =="rankgpt-api":
            api_key = OPENAI_API_KEY
        else:
            api_key=ANTHROPIC_API_KEY
        
        reranker = Reranking(method=model_category, model_name=model_name, api_key=api_key)
        reranker.rank([document])

        # Print reordered contexts
        print("Reordered Contexts:")
        for context in document.reorder_contexts:
            print(f"  - {context.text}")
        print(f"✔ {model_name} passed!\n")

    except Exception as e:
        print(f"❌ {model_name} failed with error: {e}\n")

# Iterate over all models and test each one
for category, models in HF_PRE_DEFIND_MODELS.items():
    print(category, "----------")
    if category == 'flashrank-model-file' or category =="apiranker":
        continue
    for model_key, model_name in models.items():
        test_reranking_model(category, model_key)
        break