import chromadb
from langchain_chroma import Chroma
from tutoria.rag.router import embeddings
from tutoria.rag.config import settings

# Modelo de Embeddings
chroma_client = chromadb.PersistentClient(path=settings.PERSIST_DIRECTORY)

def get_vectorstore(collection_name: str) -> Chroma:
    """Retorna a interface do LangChain para uma collection específica."""
    return Chroma(
        client=chroma_client,
        collection_name=collection_name,
        embedding_function=embeddings
    )
