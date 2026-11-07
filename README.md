# RAG From Scratch: Building a Retrieval-Augmented Generation Pipeline

A hands-on implementation of a Retrieval-Augmented Generation (RAG) system using **Python, LangChain, OpenAI Embeddings, and ChromaDB**.

This project explores the fundamental components of RAG systems, from document ingestion and text preprocessing to semantic search, retrieval, and LLM-powered answer generation.

## Project Overview

Large Language Models (LLMs) can generate useful responses, but their knowledge may be limited, outdated, or disconnected from private and domain-specific information.

Retrieval-Augmented Generation (RAG) addresses this limitation by retrieving relevant information from an external knowledge source and providing it as context to an LLM before generating an answer.

The goal of this project is to understand and implement the core RAG pipeline step by step rather than relying on high-level abstractions.

## Architecture

The project follows a standard RAG architecture:

**Document Loading → Text Splitting → Embedding Generation → Vector Storage → Retrieval → Prompt Construction → LLM Generation**

## Project Structure

The notebook is organized into four main parts:

### Part 1: RAG Overview

- Load documents from a web source using `WebBaseLoader`.
- Split documents into smaller text chunks.
- Generate vector embeddings using OpenAI.
- Store embeddings in ChromaDB.
- Retrieve relevant documents.
- Generate context-aware answers using an LLM.

### Part 2: Indexing

Explore the document indexing process:

- Token counting using `tiktoken`.
- Text embedding generation.
- Cosine similarity calculation.
- Document loading and preprocessing.
- Recursive text splitting.
- Vector database creation.

### Part 3: Retrieval

Implement semantic retrieval using ChromaDB.

- Convert documents into vector representations.
- Store vectors in a vector database.
- Perform similarity-based document retrieval.
- Configure the number of retrieved documents using `k`.

### Part 4: Generation

Build the generation component of the RAG pipeline.

- Construct custom prompts using `ChatPromptTemplate`.
- Pass retrieved documents as contextual information.
- Generate answers using OpenAI's Chat Models.
- Combine retrieval and generation into a unified LangChain chain.
- Parse the generated responses using `StrOutputParser`.

## Technologies & Libraries

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| LangChain | RAG pipeline orchestration |
| OpenAI API | Language model and embeddings |
| ChromaDB | Vector database |
| OpenAI Embeddings | Semantic text representation |
| tiktoken | Tokenization and token counting |
| NumPy | Cosine similarity calculation |
| BeautifulSoup | HTML document parsing |
| LangSmith | Optional tracing and monitoring |

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/rag-from-scratch.git
cd rag-from-scratch
```

Install the required dependencies:

```bash
pip install langchain langchain-community langchain-openai langchainhub chromadb tiktoken beautifulsoup4 numpy
```

## Configuration

Set your OpenAI API key:

```python
import os

os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"
```

Optionally, configure LangSmith for tracing and monitoring:

```python
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "YOUR_LANGSMITH_API_KEY"
```

**Security:** Never commit API keys or other sensitive credentials to GitHub. Use environment variables or a `.env` file excluded through `.gitignore`.

## Running the Project

1. Install the dependencies.
2. Configure your OpenAI API key.
3. Open `rag_from_scratch_1_to_4.ipynb` using Jupyter Notebook or Google Colab.
4. Execute the notebook cells sequentially.
5. Explore the indexing, retrieval, and generation components.
6. Test the RAG pipeline with different questions.

## Example

The project uses an article about AI agents as its document source:

[Planning and Building Autonomous Agents – Lilian Weng](https://lilianweng.github.io/posts/2023-06-23-agent/)

Example question:

```python
question = "What is Task Decomposition?"

answer = rag_chain.invoke(question)

print(answer)
```

The system retrieves relevant text chunks from the indexed article and uses them as context to generate an answer.

## Key Concepts Learned

- Understanding the architecture of RAG systems.
- Document ingestion and preprocessing.
- Tokenization and text chunking strategies.
- Dense vector embeddings.
- Cosine similarity and semantic search.
- Vector database indexing and retrieval.
- Context-aware LLM generation.
- LangChain Expression Language (LCEL).
- Building retrieval and generation chains.

## Future Improvements

Potential extensions to this implementation include:

- [ ] Support for PDF and multiple document formats.
- [ ] Persistent vector database storage.
- [ ] Advanced chunking strategies.
- [ ] Metadata filtering.
- [ ] Hybrid search (BM25 + Vector Search).
- [ ] Reranking retrieved documents.
- [ ] Conversational RAG with chat history.
- [ ] Retrieval evaluation
