import os
import re
from dotenv import load_dotenv
from groq import Groq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

load_dotenv()

DB_PATH     = "./chroma_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL  = "llama-3.3-70b-versatile"

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def load_vectorstore() -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )

STRUCTURE_KEYWORDS = ["structure", "dossier", "arborescence", "organisation", "architecture", "fichier", "module", "répertoire"]
GOAL_KEYWORDS      = ["but", "objectif", "quoi", "projet", "sert", "utilité", "rôle", "résumé", "présente", "explique"]

def is_structure_question(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in STRUCTURE_KEYWORDS)

def is_goal_question(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in GOAL_KEYWORDS)

def extract_filename(question: str) -> str | None:
    """Détecte si la question cible un fichier précis (ex: requirements.txt, graph.py)."""
    match = re.search(r'[\w\-]+\.\w+', question)
    return match.group(0).lower() if match else None

def retrieve(vectorstore: Chroma, question: str, k: int = 12) -> list:
    # --- Solution 3 : filtre direct si un nom de fichier est détecté ---
    filename = extract_filename(question)
    if filename:
        # ChromaDB retourne 10 docs par défaut — on fetch tout avec le bon limit
        total = vectorstore._collection.count()
        raw = vectorstore.get(limit=max(total, 1))
        matched = [
            Document(page_content=doc, metadata=meta)
            for doc, meta in zip(raw["documents"], raw["metadatas"])
            if filename in meta.get("source", "").lower()
        ]
        if matched:
            return matched  # résultat garanti, bypass vectorielle

    # --- Solution 1 (reranking) + garantie README ---
    results = vectorstore.similarity_search_with_score(question, k=k)

    def score_chunk(item):
        doc, score = item
        src = doc.metadata.get("source", "").lower()
        bonus = 0
        if "readme" in src:
            bonus += 0.4
        if is_structure_question(question) or is_goal_question(question):
            if "main" in src:        bonus += 0.3
            if "config" in src:      bonus += 0.25
            if "schema" in src:      bonus += 0.2
            if "docker" in src:      bonus += 0.2
            if "pyproject" in src:   bonus += 0.2
            if "agent" in src:       bonus += 0.15
        return score - bonus

    results.sort(key=score_chunk)
    top = [doc for doc, _ in results[:8]]

    if is_structure_question(question) or is_goal_question(question):
        sources_present = {d.metadata.get("source", "") for d in top}
        for doc, _ in results:
            if "readme" in doc.metadata.get("source", "").lower() and doc.metadata["source"] not in sources_present:
                top.append(doc)
                break

    return top[:8]

def generate(question: str, chunks: list, history: list = []) -> str:
    context = "\n\n---\n\n".join([
        f"Fichier: {c.metadata['source']}\n{c.page_content}"
        for c in chunks
    ])

    # Historique des 4 derniers échanges
    history_text = ""
    if history:
        last_turns = history[-8:]  # 4 questions + 4 réponses
        history_text = "\n".join([
            f"{'Utilisateur' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in last_turns
        ])

    # Instruction spéciale selon le type de question
    extra_instruction = ""
    if is_structure_question(question):
        extra_instruction = "\n- La question porte sur la STRUCTURE. Liste précisément les dossiers/fichiers visibles dans le contexte avec le rôle de chacun. Ne génère pas de structure inventée."
    elif is_goal_question(question):
        extra_instruction = "\n- La question porte sur le BUT du projet. Déduis-le du README et du code. Donne une réponse directe en 3-5 phrases."

    prompt = f"""Tu es un expert en analyse de code et architecture logicielle.
Le repo analysé est un projet Python/backend. Analyse le code fourni avec précision.

Instructions :
- Réponds toujours en français, de manière claire et précise
- Base-toi UNIQUEMENT sur le code et contexte fournis
- Cite les fichiers sources quand c'est pertinent
- Si l'info n'est pas dans le contexte, dis-le clairement sans inventer{extra_instruction}

{f"HISTORIQUE DE LA CONVERSATION:{chr(10)}{history_text}{chr(10)}" if history_text else ""}
CONTEXTE (extraits de code du repo):
{context}

QUESTION: {question}

RÉPONSE:"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=1500,
    )
    return response.choices[0].message.content

def ask(question: str, history: list = []) -> dict:
    vectorstore = load_vectorstore()
    chunks      = retrieve(vectorstore, question)
    answer      = generate(question, chunks, history)
    sources     = list({c.metadata["source"] for c in chunks})
    return {"answer": answer, "sources": sources}

if __name__ == "__main__":
    history = []
    print("GitHub RAG - tape 'quit' pour quitter\n")
    while True:
        q = input("Question : ").strip()
        if q.lower() in ("quit", "exit"):
            break
        result = ask(q, history)
        print("\nRéponse :\n", result["answer"])
        print("\nSources :", result["sources"], "\n")
        history.append({"role": "user", "content": q})
        history.append({"role": "assistant", "content": result["answer"]})
