from typing import Dict, Any, List
from langchain_core.documents import Document
from graph.state import GraphState
from ingestion import retriever


def retrieve(state: GraphState) -> Dict[str, Any]:
    """
    This node:
    1. Extracts the user's question from the current state
    2. Uses the vector store's semantic search capabilities to find relevant documents
    3. Updates the state with the retrieved documents
    
    The retriever is expected to be configured with a local vector store containing
    all necessary embeddings.
    
    Args:
        state (GraphState): The current state containing the user's question
        
    Returns:
        Dict[str, Any]: Updated state with retrieved documents and original question
    """
    print("---RETRIEVE---")
    question = state["question"]
    
    if "documents" not in state:
        print(f"[RETRIEVE] Buscando documentos para: {question}")
        documents = retriever.invoke(question)
        print(f"[RETRIEVE] Documentos encontrados: {len(documents)}")
    else:
        documents = state["documents"]

    return {"documents": documents, "question": question}
