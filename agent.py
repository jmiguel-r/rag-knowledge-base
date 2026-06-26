"""
agent.py — Agentic RAG v2 with LangGraph (100% local, zero cost)
Adaptive retrieval loop: retrieve → grade → [rewrite → retrieve]* → generate
Usage: python agent.py
"""

from __future__ import annotations

from typing import List, TypedDict

from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.graph import END, START, StateGraph
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

console = Console()

# ── Same constants as rag.py ──────────────────────────────────────────────────
CHROMA_PATH = "./chroma_db"
COLLECTION  = "knowledge_base"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL   = "llama3.2"
MAX_ITER    = 3


# ── Agent state ───────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    query:      str
    documents:  List[Document]
    answer:     str
    iterations: int
    relevant:   bool


# ── Agent class ───────────────────────────────────────────────────────────────

class AgenticRAG:
    """LangGraph-based adaptive RAG agent with query rewriting and document grading."""

    def __init__(self) -> None:
        console.print("[dim]Initializing components…[/dim]")
        embeddings  = OllamaEmbeddings(model=EMBED_MODEL)
        vectorstore = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings,
            collection_name=COLLECTION,
        )
        self.retriever = vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )
        self.llm = ChatOllama(model=LLM_MODEL, temperature=0)
        self.app = self._compile()

    # ── Nodes ─────────────────────────────────────────────────────────────────

    def rewrite_query(self, state: AgentState) -> AgentState:
        """Rewrite the query with LLM to improve retrieval on retry."""
        console.print(
            f"  [yellow]↻ Rewriting query "
            f"(attempt {state['iterations'] + 1}/{MAX_ITER})…[/yellow]"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a search query optimizer. Rewrite the user's query to be more "
             "specific and likely to retrieve relevant information from a knowledge base. "
             "Return ONLY the rewritten query, no explanation."),
            ("human", "Original query: {query}"),
        ])
        chain     = prompt | self.llm | StrOutputParser()
        new_query = chain.invoke({"query": state["query"]}).strip()
        console.print(f"  [dim]Rewritten: {new_query}[/dim]")
        return {**state, "query": new_query, "iterations": state["iterations"] + 1}

    def retrieve(self, state: AgentState) -> AgentState:
        """Retrieve document chunks from ChromaDB."""
        console.print("  [cyan]→ Retrieving from knowledge base…[/cyan]")
        docs = self.retriever.invoke(state["query"])
        console.print(f"  [dim]{len(docs)} chunks retrieved[/dim]")
        return {**state, "documents": docs}

    def grade_documents(self, state: AgentState) -> AgentState:
        """Grade each document chunk for relevance using the LLM."""
        console.print("  [cyan]→ Grading relevance…[/cyan]")
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a relevance grader. Respond ONLY with 'yes' or 'no'. "
             "Is the document chunk relevant to answering the question?"),
            ("human", "Question: {query}\n\nDocument chunk:\n{content}"),
        ])
        chain = prompt | self.llm | StrOutputParser()

        relevant_docs: List[Document] = []
        for doc in state["documents"]:
            score = chain.invoke({"query": state["query"], "content": doc.page_content})
            if "yes" in score.lower():
                relevant_docs.append(doc)

        relevant = len(relevant_docs) > 0
        tag      = "[green]✓ Relevant[/green]" if relevant else "[red]✗ Not relevant[/red]"
        console.print(
            f"  {tag} — {len(relevant_docs)}/{len(state['documents'])} chunks passed grading"
        )
        return {**state, "documents": relevant_docs, "relevant": relevant}

    def generate(self, state: AgentState) -> AgentState:
        """Generate the final answer from the graded context."""
        console.print("  [cyan]→ Generating answer…[/cyan]")
        if state["documents"]:
            context = "\n\n---\n\n".join(
                f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
                for doc in state["documents"]
            )
        else:
            context = "No relevant context was found in the knowledge base."

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a knowledgeable assistant. Answer questions based ONLY on the "
             "provided context. If the information is not in the context, say so clearly.\n\n"
             "Context:\n{context}"),
            ("human", "{query}"),
        ])
        chain  = prompt | self.llm | StrOutputParser()
        answer = chain.invoke({"context": context, "query": state["query"]})
        return {**state, "answer": answer}

    # ── Conditional edge ──────────────────────────────────────────────────────

    @staticmethod
    def should_retry(state: AgentState) -> str:
        """Route to rewrite_query on bad retrieval, or generate when ready."""
        if not state["relevant"] and state["iterations"] < MAX_ITER:
            return "rewrite_query"
        return "generate"

    # ── Graph compilation ─────────────────────────────────────────────────────

    def _compile(self):
        wf = StateGraph(AgentState)

        wf.add_node("rewrite_query",   self.rewrite_query)
        wf.add_node("retrieve",        self.retrieve)
        wf.add_node("grade_documents", self.grade_documents)
        wf.add_node("generate",        self.generate)

        wf.add_edge(START,             "retrieve")
        wf.add_edge("retrieve",        "grade_documents")
        wf.add_conditional_edges(
            "grade_documents",
            self.should_retry,
            {"rewrite_query": "rewrite_query", "generate": "generate"},
        )
        wf.add_edge("rewrite_query",   "retrieve")
        wf.add_edge("generate",        END)

        return wf.compile()

    # ── Public interface ──────────────────────────────────────────────────────

    def run(self, question: str) -> AgentState:
        initial: AgentState = {
            "query":      question,
            "documents":  [],
            "answer":     "",
            "iterations": 0,
            "relevant":   False,
        }
        return self.app.invoke(initial)


# ── Interactive chat ──────────────────────────────────────────────────────────

def interactive_chat() -> None:
    console.rule("[bold cyan]🤖 Agentic RAG v2 — LangGraph + Ollama (local)")
    console.print(
        f"[dim]LLM: {LLM_MODEL}  |  Embeddings: {EMBED_MODEL}  |  Max retries: {MAX_ITER}[/dim]"
    )
    console.print("[dim]Type your question or 'exit' to quit[/dim]\n")

    agent = AgenticRAG()
    console.print()

    while True:
        try:
            question = console.input("[bold green]You:[/bold green] ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit", "q"):
            console.print("[dim]Goodbye![/dim]")
            break

        console.print("[dim]─── Agent pipeline ──────────────────────────[/dim]")
        result = agent.run(question)
        console.print("[dim]─────────────────────────────────────────────[/dim]")

        sources = list({doc.metadata.get("source", "—") for doc in result["documents"]})

        console.print(Panel(
            Markdown(result["answer"]),
            title="[bold blue]🤖 Assistant v2[/bold blue]",
            border_style="blue",
        ))
        if sources:
            console.print(f"[dim]📎 Sources: {', '.join(sources)}[/dim]")
        console.print(
            f"[dim]↺ Retries: {result['iterations']}  |  "
            f"Relevant: {'yes' if result['relevant'] else 'no'}[/dim]\n"
        )


if __name__ == "__main__":
    interactive_chat()
