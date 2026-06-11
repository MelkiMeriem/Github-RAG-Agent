from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from core.config import DB_PATH, EMBED_MODEL

_vectorstore = None

def get_vectorstore() -> Chroma:
    """Singleton — charge le vectorstore une seule fois par session."""
    global _vectorstore
    if _vectorstore is None:
        embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        _vectorstore = Chroma(
            persist_directory=DB_PATH,
            embedding_function=embeddings
        )
    return _vectorstore

def reset_vectorstore():
    """Appeler après une nouvelle ingestion."""
    global _vectorstore
    _vectorstore = None
