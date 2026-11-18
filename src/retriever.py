import chromadb
from langchain_chroma import Chroma

from router import embeddings
from config import settings

# PersistentClient recebe um caminho absoluto para não depender do diretório de execução.
chroma_client = chromadb.PersistentClient(path=settings.PERSIST_DIRECTORY)


def get_vectorstore(collection_name: str) -> Chroma:
    """Retorna a interface LangChain para uma collection específica."""
    return Chroma(
        client=chroma_client,
        collection_name=collection_name,
        embedding_function=embeddings,
    )
