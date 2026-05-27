"""
ingest.py — Document ingestion pipeline
Loads URLs, splits into chunks, generates embeddings, and stores in ChromaDB.
Usage: python ingest.py
"""

from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from rich.console import Console

console = Console()

# ──────────────────────────────────────────
# Define the URLs that form your knowledge base
# ──────────────────────────────────────────
URLS = [
    "https://aiostrategy.tech",
    # Add more URLs here:
    # "https://aiostrategy.tech/services",
    # "https://example.com/your-page",
]

CHROMA_PATH = "./chroma_db"
COLLECTION  = "knowledge_base"
EMBED_MODEL = "nomic-embed-text"  # local embedding model via Ollama


def ingest():
    console.rule("[bold cyan]🔄 RAG Document Ingestion (Ollama)")

    # Step 1 — Load web pages
    console.print(f"\n[yellow]Loading {len(URLS)} URL(s)...[/yellow]")
    loader = WebBaseLoader(URLS)
    docs = loader.load()
    console.print(f"[green]✓ {len(docs)} document(s) loaded[/green]")

    # Step 2 — Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,      # characters per chunk
        chunk_overlap=150,    # overlap to preserve context between chunks
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(docs)
    console.print(f"[green]✓ {len(chunks)} chunks generated[/green]")

    # Step 3 — Generate embeddings and store in ChromaDB
    console.print(f"\n[yellow]Generating embeddings with Ollama ({EMBED_MODEL})...[/yellow]")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
        collection_name=COLLECTION,
    )

    console.print(f"[green]✓ Vector store saved to [bold]{CHROMA_PATH}[/bold][/green]")
    console.print(f"[green]✓ Total vectors indexed: {vectorstore._collection.count()}[/green]")
    console.rule("[bold green]✅ Ingestion complete")


if __name__ == "__main__":
    ingest()
