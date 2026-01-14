"""
RankifyAgent Usage Examples

This script demonstrates 4 different ways to use the RankifyAgent:
1. Quick programmatic recommendation
2. Conversational interface with Azure OpenAI
3. Constraint-based recommendations
4. Getting all available models

Run: python examples/agent_examples.py
"""

import os

# Set Azure environment variables (replace with your values)
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://openaireceiptwestus.openai.azure.com/"
os.environ["AZURE_OPENAI_API_KEY"] = "your-api-key-here"
os.environ["AZURE_DEPLOYMENT_NAME"] = "gpt-4o"
os.environ["AZURE_API_VERSION"] = "2024-05-01-preview"


def example_1_quick_recommendation():
    """
    Example 1: Quick Programmatic Recommendation
    
    Use the `recommend()` function for instant model recommendations.
    """
    print("=" * 60)
    print("Example 1: Quick Programmatic Recommendation")
    print("=" * 60)
    
    from rankify.agent import recommend
    
    # Get recommendation for QA task with GPU
    result = recommend(task="qa", gpu=True, include_rag=True)
    
    print(f"\n📦 Recommended Retriever: {result.retriever.name}")
    print(f"   Method: {result.retriever.method}")
    print(f"   Speed: {result.retriever.speed.value}")
    print(f"   GPU Required: {result.retriever.gpu_required}")
    
    if result.reranker:
        print(f"\n🔄 Recommended Reranker: {result.reranker.name}")
        print(f"   Method: {result.reranker.method}")
        print(f"   Accuracy: {result.reranker.accuracy.value}")
    
    if result.rag_method:
        print(f"\n🧠 Recommended RAG Method: {result.rag_method.name}")
        print(f"   Method: {result.rag_method.method}")
    
    print("\n📝 Explanation:")
    print(result.explanation)
    

def example_2_conversational_agent():
    """
    Example 2: Conversational Interface with Azure OpenAI
    
    Have a natural conversation to get model recommendations.
    """
    print("\n" + "=" * 60)
    print("Example 2: Conversational Interface")
    print("=" * 60)
    
    from rankify.agent import RankifyAgent
    
    # Initialize agent with Azure backend
    agent = RankifyAgent(backend="azure")
    
    # First message
    response = agent.chat(
        "I need to build a document search system for legal documents. "
        "I have a GPU available and need high accuracy."
    )
    
    print(f"\n🤖 Agent Response:\n{response.message}")
    
    if response.code_snippet:
        print(f"\n💻 Generated Code:\n```python\n{response.code_snippet}\n```")
    
    # Follow-up question
    response2 = agent.chat(
        "What if I want to deploy this to production with low latency?"
    )
    
    print(f"\n🤖 Agent Response (follow-up):\n{response2.message}")


def example_3_constraint_based():
    """
    Example 3: Constraint-Based Recommendations
    
    Specify constraints like GPU availability, latency, API usage.
    """
    print("\n" + "=" * 60)
    print("Example 3: Constraint-Based Recommendations")
    print("=" * 60)
    
    from rankify.agent import RankifyRecommender
    
    recommender = RankifyRecommender()
    
    # Scenario A: No GPU, need fast inference
    print("\n📌 Scenario A: No GPU, Fast Inference Needed")
    result_a = recommender.recommend(
        task="search",
        constraints={"gpu": False, "prefer_speed": True},
        include_reranker=True,
    )
    print(f"   Retriever: {result_a.retriever.name} (Speed: {result_a.retriever.speed.value})")
    if result_a.reranker:
        print(f"   Reranker: {result_a.reranker.name}")
    
    # Scenario B: GPU available, highest accuracy
    print("\n📌 Scenario B: GPU Available, Highest Accuracy")
    result_b = recommender.recommend_for_accuracy(task="qa", gpu_available=True)
    print(f"   Retriever: {result_b.retriever.name} (Accuracy: {result_b.retriever.accuracy.value})")
    if result_b.reranker:
        print(f"   Reranker: {result_b.reranker.name}")
    
    # Scenario C: Production deployment
    print("\n📌 Scenario C: Production Deployment (Fast + Reliable)")
    result_c = recommender.recommend_for_production(gpu_available=False, api_allowed=True)
    print(f"   Retriever: {result_c.retriever.name}")
    if result_c.reranker:
        print(f"   Reranker: {result_c.reranker.name} (API: {result_c.reranker.api_provider})")


def example_4_list_models():
    """
    Example 4: Get All Available Models
    
    Explore what models are available in each category.
    """
    print("\n" + "=" * 60)
    print("Example 4: Available Models")
    print("=" * 60)
    
    from rankify.agent import (
        get_all_retrievers,
        get_all_rerankers,
        get_all_rag_methods,
    )
    
    retrievers = get_all_retrievers()
    rerankers = get_all_rerankers()
    rag_methods = get_all_rag_methods()
    
    print(f"\n📦 Retrievers ({len(retrievers)} available):")
    for name, model in list(retrievers.items())[:5]:
        print(f"   - {model.name}: {model.description[:50]}...")
    
    print(f"\n🔄 Rerankers ({len(rerankers)} available):")
    for name, model in list(rerankers.items())[:5]:
        print(f"   - {model.name}: {model.speed.value}, {model.accuracy.value}")
    
    print(f"\n🧠 RAG Methods ({len(rag_methods)} available):")
    for name, model in rag_methods.items():
        print(f"   - {model.name}: {model.description[:50]}...")


if __name__ == "__main__":
    # Run examples (comment out agent example if no API key)
    example_1_quick_recommendation()
    example_3_constraint_based()
    example_4_list_models()
    
    # Uncomment to test conversational agent (requires API key)
    # example_2_conversational_agent()
