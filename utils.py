from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()  # Load from .env into os.environ

def init_llm():
    return ChatGoogleGenerativeAI(model="gemini-2.0-flash")

def init_embeddings_and_vector_store():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = InMemoryVectorStore(embedding=embeddings)
    return vector_store
