"""
rag.py — RAG query engine with Ollama (100% local, zero cost)
Retrieves relevant context from ChromaDB and generates answers with a local LLM.
Usage: python rag.py
"""

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt

console = Console()

CHROMA_PATH = "./chroma_db"
COLLECTION  = "knowledge_base"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL   = "llama3.2"

# ──────────────────────────────────────────
# Language options
# ──────────────────────────────────────────
LANGUAGES = {
    "1": ("English",  "Always respond in English, regardless of the language of the context or the question."),
    "2": ("Spanish",  "Responde siempre en español, sin importar el idioma del contexto o de la pregunta."),
    "3": ("Auto",     "Respond in the same language the user writes in."),
}

SYSTEM_PROMPT_BASE = """You are a knowledgeable assistant that answers questions based
ONLY on the provided context. If the information is not available in the context,
clearly state that you don't have that information in your knowledge base.

Retrieved context:
{context}

{language_instruction}
"""


def pick_language() -> tuple[str, str]:
    """Ask the user to pick a response language at startup."""
    console.print("\n[bold]🌐 Select response language:[/bold]")
    for key, (name, _) in LANGUAGES.items():
        console.print(f"  [cyan]{key}[/cyan]  {name}")

    choice = Prompt.ask("\nChoice", choices=list(LANGUAGES.keys()), default="1")
    name, instruction = LANGUAGES[choice]
    console.print(f"[green]✓ Language set to: [bold]{name}[/bold][/green]\n")
    return name, instruction


def build_prompt(language_instruction: str) -> ChatPromptTemplate:
    """Build the prompt template with the selected language instruction."""
    system = SYSTEM_PROMPT_BASE.format(
        context="{context}",
        language_instruction=language_instruction,
    )
    return ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "{question}"),
    ])


def format_docs(docs: list) -> str:
    """Concatenate retrieved document chunks into a single context string."""
    return "\n\n---\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in docs
    )


def build_rag_chain(language_instruction: str, k: int = 4):
    """Build and return the full RAG chain using LCEL."""
    embeddings  = OllamaEmbeddings(model=EMBED_MODEL)
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
        collection_name=COLLECTION,
    )
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
    llm            = ChatOllama(model=LLM_MODEL, temperature=0)
    prompt         = build_prompt(language_instruction)

    # LCEL chain: retrieve → format → prompt → LLM → parse
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever


def query(question: str, chain, retriever) -> dict:
    """Run a RAG query and return the answer with its sources."""
    docs    = retriever.invoke(question)
    sources = list({doc.metadata.get("source", "—") for doc in docs})
    answer  = chain.invoke(question)
    return {"answer": answer, "sources": sources}


def interactive_chat():
    """Interactive terminal chat loop."""
    console.rule("[bold cyan]🤖 RAG Knowledge Base Assistant — Ollama (local)")
    console.print(f"[dim]LLM: {LLM_MODEL}  |  Embeddings: {EMBED_MODEL}[/dim]")

    # Language selection at startup
    lang_name, lang_instruction = pick_language()

    console.print("[dim]Type your question or 'exit' to quit  |  '/lang' to change language[/dim]\n")

    chain, retriever = build_rag_chain(lang_instruction)

    while True:
        try:
            question = console.input("[bold green]You:[/bold green] ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not question:
            continue

        # Exit
        if question.lower() in ("exit", "quit", "q"):
            console.print("[dim]Goodbye![/dim]")
            break

        # Change language mid-session
        if question.lower() == "/lang":
            lang_name, lang_instruction = pick_language()
            chain, retriever = build_rag_chain(lang_instruction)
            console.print(f"[green]✓ Language changed to [bold]{lang_name}[/bold][/green]\n")
            continue

        with console.status("[cyan]Searching knowledge base...[/cyan]"):
            result = query(question, chain, retriever)

        console.print(Panel(
            Markdown(result["answer"]),
            title=f"[bold blue]🤖 Assistant[/bold blue] [dim]({lang_name})[/dim]",
            border_style="blue",
        ))
        console.print(f"[dim]📎 Sources: {', '.join(result['sources'])}[/dim]\n")


if __name__ == "__main__":
    interactive_chat()
