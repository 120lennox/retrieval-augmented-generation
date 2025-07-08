# Simplified RAG Tutorial - Step by Step

## What is RAG?
**Retrieval Augmented Generation (RAG)** = Search + Generate
- **Search**: Find relevant information from documents
- **Generate**: Use that information to answer questions

Think of it like an open-book exam where you can look up answers before writing them.

## The Big Picture
RAG has 2 main parts:
1. **Indexing** (done once): Prepare your documents for searching
2. **Retrieval + Generation** (done each time): Find info and create answers

---

## Part 1: Setup Your Tools

### 1.1 Install Required Packages
```bash
pip install langchain-openai langchain-community langgraph langchain-text-splitters
```

### 1.2 Set Up Your Components
```python
# Import everything you need
import getpass
import os
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

# Get your OpenAI API key
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

# Set up the 3 core components
llm = init_chat_model("gpt-4o-mini", model_provider="openai")  # The answerer
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")  # The converter
vector_store = InMemoryVectorStore(embeddings)  # The database
```

**What each component does:**
- `llm`: Generates human-like answers
- `embeddings`: Converts text to numbers for searching
- `vector_store`: Stores and searches through document embeddings

---

## Part 2: Indexing (Prepare Your Documents)

### 2.1 Load a Document
```python
import bs4
from langchain_community.document_loaders import WebBaseLoader

# Load a web page (you can use any document)
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

print(f"Loaded document with {len(docs[0].page_content)} characters")
```

### 2.2 Split the Document into Chunks
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Split the large document into smaller pieces
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # Each chunk is max 1000 characters
    chunk_overlap=200,    # 200 characters overlap between chunks
)
all_splits = text_splitter.split_documents(docs)

print(f"Split into {len(all_splits)} chunks")
```

**Why split?** Large documents are hard to search through and won't fit in the AI model's memory.

### 2.3 Store the Chunks
```python
# Convert chunks to embeddings and store them
document_ids = vector_store.add_documents(documents=all_splits)
print(f"Stored {len(document_ids)} document chunks")
```

**What happened?** Your document chunks are now stored as searchable vectors!

---

## Part 3: Retrieval + Generation (The RAG Chain)

### 3.1 Set Up the Application Structure
```python
from langchain import hub
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict

# Get a pre-made prompt for RAG
prompt = hub.pull("rlm/rag-prompt")

# Define what information flows through your app
class State(TypedDict):
    question: str           # User's question
    context: List[Document] # Retrieved documents
    answer: str            # Generated answer
```

### 3.2 Create the Two Main Functions
```python
def retrieve(state: State):
    """Find relevant documents for the question"""
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

def generate(state: State):
    """Generate an answer using the retrieved documents"""
    # Combine all retrieved documents into one text
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    
    # Create a prompt with the question and context
    messages = prompt.invoke({
        "question": state["question"], 
        "context": docs_content
    })
    
    # Get the AI's response
    response = llm.invoke(messages)
    return {"answer": response.content}
```

### 3.3 Connect Everything Together
```python
# Build the application flow: retrieve → generate
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()
```

---

## Part 4: Test Your RAG System

### 4.1 Ask a Question
```python
# Test your RAG system
result = graph.invoke({"question": "What is Task Decomposition?"})

print("ANSWER:")
print(result["answer"])
print("\nSOURCE DOCUMENTS:")
for i, doc in enumerate(result["context"]):
    print(f"{i+1}. {doc.page_content[:200]}...")
```

### 4.2 Stream the Response (Optional)
```python
# See each step as it happens
for step in graph.stream(
    {"question": "What is Task Decomposition?"}, 
    stream_mode="updates"
):
    print(f"Step: {step}")
```

---

## Complete Working Example

Here's everything together in one block:

```python
# Setup
import getpass
import os
import bs4
from langchain import hub
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict

# Get API key
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

# Initialize components
llm = init_chat_model("gpt-4o-mini", model_provider="openai")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = InMemoryVectorStore(embeddings)

# Load and process documents
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=("post-content", "post-title", "post-header")))
)
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)
vector_store.add_documents(documents=all_splits)

# Define application
prompt = hub.pull("rlm/rag-prompt")

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

# Build and test
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

# Test it!
result = graph.invoke({"question": "What is Task Decomposition?"})
print(result["answer"])
```

---

## Key Concepts Summary

1. **Embeddings**: Convert text to numbers that capture meaning
2. **Vector Store**: Database that stores and searches embeddings
3. **Chunking**: Breaking large documents into smaller, searchable pieces
4. **Retrieval**: Finding relevant document chunks for a question
5. **Generation**: Using retrieved context to create an answer
6. **LangGraph**: Framework that connects retrieval and generation steps

---

## Next Steps

Once you understand this basic RAG system, you can:
- Add conversation history (Part 2 of the tutorial)
- Use different document types (PDFs, CSVs, etc.)
- Improve retrieval with query analysis
- Add custom prompts
- Use different vector stores (Chroma, Pinecone, etc.)

The key is understanding this flow: **Load → Split → Store → Retrieve → Generate**