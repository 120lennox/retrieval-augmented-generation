"""
Description: This is the improved version for bot1. Bot 2 remembers stores chat history. 
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
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict

# set environmental variables
os.environ["LANGSMTH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "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"
os.environ["USER_AGENT"] = "RAG Bot/V2"

def get_github_token():
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        token = getpass.getpass("Enter your github acess token")
        os.environ("GITHUB_TOKEN") = token

github_token = get_github_token()

# initialize the rag system

def initialize_rag_system():
    try:
        # LLM config
        llm = ChatOpenAI(
            model = "openai/gpt-4.1",
            openai_api_base="https://models.github.ai/inference",
            openai_api_key = github_token,
            temperature = 0.7
        )

        # initialize embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
        )

        # init vectors 
        vectore_store = InMemoryVectorStore(embeddings)

        loader = WebBaseLoader(
            web_path = "https://lilianweng.github.io/posts/2023-06-23-agent/",
            bs_kwargs = dict(
                parse_only=bs4.SoupStrainer(
                    class_=("post-content", "post-title", "post-loader")
                )
            ),
        )

        docs = loader.load()


        # split chunk of data
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 200,
        )

        all_splits = text_splitter.split_documents(docs)

        # store splitted docs as in vectore storage
        document_id = vectore_store.add_documents(documents=all_splits)

        # error checking console output
        print(f"loaded a document with {len(docs[0].page_content)} characters. \n Added {len(document_id)} document chunks in a vector store.")

        return llm, vectore_store
    
    except Exception as e:
        print(f"Error initializing RAG system {e}")
        raise

class State(TypedDict):
    question: str
    chat_history: List[tuple]
    context: List[Document]
    answer: str

