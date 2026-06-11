from langgraph.graph import StateGraph, END
from rag.state import RAGState
from rag.nodes import query_rewriter, retriever, grader, retry_rewriter, generator


def route_after_grade(state: RAGState) -> str:
    """Routing conditionnel après le grader."""
    if state["grade"] == "relevant":
        return "generator"
    if state.get("retry_count", 0) >= 2:
        return "generator"  # on génère quand même après 2 essais
    return "retry_rewriter"


def build_graph() -> StateGraph:
    graph = StateGraph(RAGState)

    # Nodes
    graph.add_node("query_rewriter",  query_rewriter)
    graph.add_node("retriever",       retriever)
    graph.add_node("grader",          grader)
    graph.add_node("retry_rewriter",  retry_rewriter)
    graph.add_node("generator",       generator)

    # Edges
    graph.set_entry_point("query_rewriter")
    graph.add_edge("query_rewriter", "retriever")
    graph.add_edge("retriever",      "grader")
    graph.add_conditional_edges(
        "grader",
        route_after_grade,
        {
            "generator":      "generator",
            "retry_rewriter": "retry_rewriter",
        }
    )
    graph.add_edge("retry_rewriter", "retriever")
    graph.add_edge("generator",      END)

    return graph.compile()


# Singleton
_graph = None

def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def ask(question: str, history: list = []) -> dict:
    """Point d'entrée public — remplace l'ancien ask() de query.py."""
    graph = get_graph()
    result = graph.invoke({
        "question":    question,
        "history":     history,
        "retry_count": 0,
        "grade":       "",
        "rewritten_query": "",
        "chunks":      [],
        "answer":      "",
        "sources":     [],
    })
    return {
        "answer":  result["answer"],
        "sources": result["sources"],
    }
