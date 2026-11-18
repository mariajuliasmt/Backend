import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.retriever import get_vectorstore
from src.router import select_target_collections
from src.config import settings


# =============== CRIAÇÃO DO BANCO DE DADOS ===============
def run_workflow_1():

    print("Iniciando Criação DB...")
    loader = PyPDFDirectoryLoader(settings.BASE_FOLDER,glob="*.pdf")
    documents = loader.load()

    if not documents:
        print("Nenhum PDF encontrado.")
        return

    docs_by_file = {}

    for doc in documents:
        source = doc.metadata["source"]
        docs_by_file.setdefault(
            source,
            []
        ).append(doc)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_SIZE,
        chunk_overlap=settings.overlap,
        add_start_index=True
    )

    for source, file_docs in docs_by_file.items():

        filename = os.path.basename(source)
        print(f"\nProcessando: {filename}")
        chunks = splitter.split_documents(file_docs)
        grouped_chunks = {
            "estagio_1": [],
            "estagio_2": [],
            "estagio_3": []
        }

        for i in range(len(chunks)):
            start = max(0,i - settings.WINDOW_SIZE // 2)
            end = min(
                len(chunks),
                i + settings.WINDOW_SIZE // 2 + 1
            )
            context_text = "\n".join(
                chunk.page_content
                for chunk in chunks[start:end]
            )
            chunk = chunks[i]
            chunk.metadata["source_file"] = filename

            target_collections, scores = select_target_collections(
            context_text, settings.DUPLICATION_THRESHOLD)

            for collection in target_collections:
                chunk.metadata["stage"] = collection
                grouped_chunks[collection].append(chunk)

        for stage, docs in grouped_chunks.items():
            if not docs:
                continue
            vectorstore = get_vectorstore(stage)
            vectorstore.add_documents(docs)
            print(f"{len(docs)} chunks enviados para {stage}")

    print("\nCriação concluída.")

if __name__ == "__main__":
    run_workflow_1()
