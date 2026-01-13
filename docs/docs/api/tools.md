# Tools Module

The tools module provides utilities for web search and other agent-based capabilities.

## WebSearchTool

The `WebSearchTool` enables real-time web search capabilities for RAG applications.

### Features
- Integrates with SERP providers (e.g., Serper.dev)
- Automatic content extraction and context building
- Async-compatible design

### Usage

```python
from rankify.tools.Tools import WebSearchTool

# Initialize the tool
search_tool = WebSearchTool(
    search_provider='serper',
    search_provider_api_key='your-api-key'
)

# Setup (loads necessary components)
search_tool.setup()

# Perform search
results = search_tool.forward("What is RAG in NLP?", num_result=10)
print(results)
```

## API Reference

### Tool Base Class
::: rankify.tools.Tools.Tool
    handler: python
    options:
        show_source: true
        heading_level: 3

### WebSearchTool
::: rankify.tools.Tools.WebSearchTool
    handler: python
    options:
        show_source: true
        heading_level: 3
