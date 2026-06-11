from typing import TypedDict, List, Optional
from langchain_core.documents import Document

class RAGState(TypedDict):
    # Input
    question: str
    history: List[dict]

    # Processing
    rewritten_query: str
    chunks: List[Document]
    grade: str             # "relevant" | "retry" | "give_up"
    retry_count: int

    # Output
    answer: str
    sources: List[str]
