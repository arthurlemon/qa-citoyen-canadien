---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.4
  kernelspec:
    display_name: .venv
    language: python
    name: python3
---

```python
import os
from haystack import Pipeline, Document
from haystack.utils import Secret
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever, InMemoryEmbeddingRetriever
from haystack.components.generators import OpenAIGenerator
from haystack.components.builders.answer_builder import AnswerBuilder
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.converters import PyPDFToDocument
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.components.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
from langfuse.decorators import observe
from dotenv import load_dotenv
load_dotenv(override=True)
```

### Naive documents

```python
# Write documents to InMemoryDocumentStore
document_store = InMemoryDocumentStore()
document_store.write_documents([
    Document(content="My name is Jean and I live in Paris."), 
    Document(content="My name is Mark and I live in Berlin."), 
    Document(content="My name is Giorgio and I live in Rome.")
])

# Build a RAG pipeline
prompt_template = """
Given these documents, answer the question.
Documents:
{% for doc in documents %}
    {{ doc.content }}
{% endfor %}
Question: {{question}}
Answer:
"""

retriever = InMemoryBM25Retriever(document_store=document_store)
prompt_builder = PromptBuilder(template=prompt_template)
llm = OpenAIGenerator(model="gpt-4o-mini", api_key=Secret.from_token(os.getenv("OPENAI_API_KEY")))

rag_pipeline = Pipeline()
rag_pipeline.add_component("retriever", retriever)
rag_pipeline.add_component("prompt_builder", prompt_builder)
rag_pipeline.add_component("llm", llm)
rag_pipeline.connect("retriever", "prompt_builder.documents")
rag_pipeline.connect("prompt_builder", "llm")
```

```python
@observe(as_type="generation")
def ask_question(question:str) -> str:
    results = rag_pipeline.run(
        {
            "retriever": {"query": question},
            "prompt_builder": {"question": question},
        }
    )

    return (results["llm"]["replies"])
```

```python
ask_question("Who lives in Paris?")
```

### PDF of interest

```python
document_store = InMemoryDocumentStore()

preprocessing_pipeline = Pipeline()
preprocessing_pipeline.add_component("converter", PyPDFToDocument())
preprocessing_pipeline.add_component("cleaner", DocumentCleaner())
preprocessing_pipeline.add_component("splitter", DocumentSplitter(split_by="sentence", split_length=5))
preprocessing_pipeline.add_component("embedder", OpenAIDocumentEmbedder())
preprocessing_pipeline.add_component("writer", DocumentWriter(document_store=document_store))
preprocessing_pipeline.connect("converter", "cleaner")
preprocessing_pipeline.connect("cleaner", "splitter")
preprocessing_pipeline.connect("splitter", "embedder")
preprocessing_pipeline.connect("embedder", "writer")
```

```python
@observe()
def ingest_pdf(pipeline, pdf_path: str) -> None:
    pipeline.run({"converter": {"sources": [pdf_path]}})
```

```python
pdf_path = "../data/decouvrir.pdf"
ingest_pdf(preprocessing_pipeline, pdf_path)
```

```python
template = """
Answer the questions based on the given context.

Context:
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Question: {{ question }}
Answer:
"""
inference_pipeline = Pipeline()
inference_pipeline.add_component("embedder", OpenAITextEmbedder())
inference_pipeline.add_component("retriever", InMemoryEmbeddingRetriever(document_store=document_store))
inference_pipeline.add_component("prompt_builder", PromptBuilder(template=template))
inference_pipeline.add_component(
    "generation",
    OpenAIGenerator(model="gpt-4o"),
)

inference_pipeline.connect("embedder.embedding", "retriever.query_embedding")
inference_pipeline.connect("retriever", "prompt_builder.documents")
inference_pipeline.connect("prompt_builder", "generation")
```

```python
docs
```

```python

```
