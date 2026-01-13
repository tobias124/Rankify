# Tools Module

The tools module provides utilities for web search and other agent-based capabilities.

## WebSearchTool

The `WebSearchTool` enables real-time web search capabilities for RAG applications.

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

::: rankify.tools.Tools
    options:
        show_source: true
        members: true
        show_root_heading: true
        heading_level: 2
        docstring_style: google
