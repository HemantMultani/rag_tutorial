from typing import Annotated, Literal, TypedDict
from langchain import hub
from langgraph.graph import START, StateGraph
from ingest import load_doc_in_vector_store
from utils import init_llm


llm = init_llm()
vector_store = load_doc_in_vector_store()
prompt = hub.pull("rlm/rag-prompt")

class Search(TypedDict):
    query: Annotated[str, ..., "Create a detailed query to search in the vector store"]  #These Metadata description will help LLM understand the structure
    section: Annotated[Literal['beginning', 'middle', 'end'], ..., "Section of the document to search in"]

class State(TypedDict):
    """This will be our State (object) for our Graph"""
    question: str
    search: Search
    context: str
    answer: str

def analyze_query(state: State):
    structured_llm = llm.with_structured_output(Search)
    search = structured_llm.invoke(state["question"])
    return {"search": search}

def retreive(state: State):
    search = state["search"]
    context = vector_store.similarity_search(
        #state["question"]
        search["query"],
        filter=lambda doc: doc.metadata.get("section") == search["section"]
    )
    return {"context": context}

def generate(state: State):
    context = state["context"]
    if not context:
        return {"answer": "No relevant context found."}
    message = prompt.invoke({
        "question": state["question"],
        "context": context
    })
    
    res = llm.invoke(message)
    return {"answer": res.content}

def rag_chain():
    graph_builder = StateGraph(State).add_sequence([analyze_query, retreive, generate])
    graph_builder.add_edge(START, "analyze_query")
    return graph_builder.compile()