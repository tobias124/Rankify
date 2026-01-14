# RankifyAgent API

Intelligent model recommendation system for Rankify.

## Overview

RankifyAgent provides:
- **Programmatic recommendations** via `recommend()` and `RankifyRecommender`
- **Conversational interface** via `RankifyAgent` class
- **Model registry** with metadata for all available models

## Quick Example

```python
from rankify.agent import recommend, RankifyAgent

# Quick recommendation
result = recommend(task="qa", gpu=True)
print(result.retriever.name)  # "BGE"

# Conversational agent
agent = RankifyAgent(backend="azure")
response = agent.chat("I need a fast search system")
print(response.message)
```

## API Reference

::: rankify.agent
    options:
        show_source: true
        members: true
        show_root_heading: true
        heading_level: 2
        docstring_style: google
