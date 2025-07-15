"""
Description: Program retrieves information from website. Provides the context to LLM. LLM generates text message in response to user prompt. Uses Retrieval Augmented Generation. 
"""

import getpass
import os
from langchain_openai import ChatOpenAI
from langchain_core.vectorstores import InMemoryVectorStore
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain import hub
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict

# Set environment variables
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "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"
os.environ["USER_AGENT"] = "RAG Bot/1.0"

def get_github_token():
    token = os.environ.get("GITHUB_ACCESS_TOKEN")
    if not token:
        token = getpass.getpass("Enter your GitHub access token: ")
        os.environ["GITHUB_ACCESS_TOKEN"] = token
    
    os.environ["OPENAI_API_KEY"] = token
    return token

# Get token securely
github_token = get_github_token()

def initialize_rag_system():
    try:
        # Configure LLM
        llm = ChatOpenAI(
            model="openai/gpt-4.1",
            openai_api_base="https://models.github.ai/inference",
            openai_api_key=github_token,
            temperature=0.7
        )
        
        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Initialize vector store
        vector_store = InMemoryVectorStore(embeddings)
        
        # Load and process documents
        loader = WebBaseLoader(
            web_path=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(
                    class_=("post-content", "post-title", "post-loader")
                )
            ),
        )
        
        docs = loader.load()
        print(f"Loaded documents with {len(docs[0].page_content)} characters")
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        all_splits = text_splitter.split_documents(docs)
        
        # Store chunks in vector store
        document_ids = vector_store.add_documents(documents=all_splits)
        print(f"Added {len(document_ids)} document chunks to vector store")
        
        # Get prompt template
        prompt = hub.pull("rlm/rag-prompt")
        
        return llm, vector_store, prompt
        
    except Exception as e:
        print(f"Error initializing RAG system: {e}")
        raise


class State(TypedDict):
    question: str  # User's question
    context: List[Document]  # Retrieved documents
    answer: str  # Generated answer

def create_rag_graph(llm, vector_store, prompt):
    
    def retrieve(state: State):
        """Retrieve relevant documents"""
        try:
            retrieved_docs = vector_store.similarity_search(
                state["question"], 
                k=4  # Limit number of results
            )
            return {"context": retrieved_docs}
        except Exception as e:
            print(f"Error in retrieve: {e}")
            return {"context": []}
    
    def generate(state: State):
        """Generate answers with the retrieved documents"""
        try:
            # Combine the retrieved documents into one
            docs_content = "\n\n".join(
                doc.page_content for doc in state["context"]
            )
            
            # Create prompt with question and context
            messages = prompt.invoke({
                "question": state["question"],
                "context": docs_content
            })
            
            # Get response from LLM
            response = llm.invoke(messages)
            
            return {"answer": response.content}
        except Exception as e:
            print(f"Error in generate: {e}")
            return {"answer": f"Error generating response: {e}"}
    
    # Build the graph
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()
    
    return graph

def main():
    """Main function to run the RAG system"""
    try:
        # Initialize system
        llm, vector_store, prompt = initialize_rag_system()
        
        # Create graph
        graph = create_rag_graph(llm, vector_store, prompt)
        
        # Interactive loop
        while True:
            user_question = input("\nEnter your question (or 'quit' to exit): ")
            
            if user_question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not user_question.strip():
                print("Please enter a valid question.")
                continue
            
            try:
                result = graph.invoke({"question": user_question})
                print(f"\nAnswer: {result['answer']}")
                
            except Exception as e:
                print(f"Error processing question: {e}")
    
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Fatal error: {e}")

if __name__ == "__main__":
    main()

# import getpass
# import os
# from langchain.chat_models import init_chat_model
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from langchain_core.vectorstores import InMemoryVectorStore
# import bs4
# from langchain_community.document_loaders import WebBaseLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain import hub
# from langchain_core.documents import Document
# from langgraph.graph import START, StateGraph
# from typing_extensions import List, TypedDict

# os.environ["LANGSMITH_TRACING"] = "true"
# os.environ["USER_AGENT"] = "RAG Bot/1.0"

# # get github access token
# if not os.environ.get("GITHUB_ACCESS_TOKEN"):
#   os.environ["GITHUB_ACCESS_TOKEN"] = "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"

# # configure LLM
# llm = ChatOpenAI(
#     model="openai/gpt-4.1",  # or whatever model name works
#     openai_api_base="https://models.github.ai/inference",
#     openai_api_key=os.environ["GITHUB_ACCESS_TOKEN"],
#     temperature=0.7
# )

# # using huggingface embeddings because github API models does not support embedding models  
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# # in memory storage (more like database)
# # stores our embeddings in memory: for faster searches and retrievals
# vector_store = InMemoryVectorStore(embeddings)

# # loading documents
# loader = WebBaseLoader(
#   web_path=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
#   bs_kwargs=dict(
#     parse_only=bs4.SoupStrainer(class_=("post-content", "post-title", "post-loader"))
#   ),
# )

# docs = loader.load()
# print(f"loaded documents with {len(docs[0].page_content)} characters")

# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size = 1000,
#     chunk_overlap=200,
# )
# all_splits = text_splitter.split_documents(docs)

# # storing the chunks
# # the document chunks have now been stored as searchable vetctors
# document_ids = vector_store.add_documents(documents=all_splits)

# prompt = hub.pull("rlm/rag-prompt")

# # define what information flows through your app
# class State (TypedDict):
#     question: str # user's question
#     context: List[Document] # retrieved documents
#     answer = str # generted answer

# # retrieve relevant documents
# def retrieve(state: State):
#     retrieved_docs = vector_store.similarity_search(state["question"])

#     return {"context": retrieved_docs}

# def generate(state: State):
#     "generate answers with the retrieved documents"
    
#     #combine the retrived documents into one
#     docs_content = "\n\n".join(doc.page_content for doc in state["context"])

#     # create prompt with question and context 
#     messages = prompt.invoke({
#         "question": state["question"],
#         "context": docs_content
#     })

#     # get response from LLM
#     response = llm.invoke(messages)

#     return {"answer": response.content}

# graph_builder = StateGraph(State).add_sequence([retrieve, generate])
# graph_builder.add_edge(START, "retrieve")
# graph = graph_builder.compile()

# prompt = input("prompt: ")
# result = graph.invoke(prompt)
# print(result.content)
