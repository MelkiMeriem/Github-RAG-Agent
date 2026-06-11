import re
from langchain_core.documents import Document
from core.config import RETRIEVAL_K, RETRIEVAL_TOP_N
from core.llm import call_llm
from core.vectorstore import get_vectorstore
from rag.state import RAGState

# ---------------------------------------------------------------------------
# Node 1 — Query Rewriter
# Résout les coréférences ("le 2ème", "ce fichier", "pareil pour X")
# ---------------------------------------------------------------------------
def query_rewriter(state: RAGState) -> RAGState:
    question = state["question"]
    history  = state.get("history", [])

    if not history:
        return {**state, "rewritten_query": question}

    history_text = "\n".join([
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content'][:300]}"
        for m in history[-6:]
    ])

    prompt = f"""Tu es un assistant qui reformule des questions ambiguës.
Voici l'historique d'une conversation et une nouvelle question.
Reformule la question pour qu'elle soit COMPLÈTE et AUTONOME :
- Remplace "ça", "ce fichier", "le premier/deuxième", "pareil", "idem", "lui", "ce projet" par leur référent explicite
- Si la question est déjà claire et autonome, retourne-la EXACTEMENT telle quelle
- Retourne UNIQUEMENT la question reformulée, sans explication ni ponctuation ajoutée

HISTORIQUE:
{history_text}

QUESTION: {question}

QUESTION REFORMULÉE:"""

    rewritten = call_llm(prompt, max_tokens=150, temperature=0)
    print(f"[Rewriter] '{question}' → '{rewritten}'")
    return {**state, "rewritten_query": rewritten}


# ---------------------------------------------------------------------------
# Node 2 — Retriever
# Cherche les chunks pertinents (filtre direct si nom de fichier détecté)
# ---------------------------------------------------------------------------
def _extract_filename(question: str) -> str | None:
    match = re.search(r'[\w\-]+\.\w+', question)
    return match.group(0).lower() if match else None

def _is_type(question: str, keywords: list[str]) -> bool:
    q = question.lower()
    return any(kw in q for kw in keywords)

STRUCTURE_KW = ["structure", "dossier", "arborescence", "architecture", "fichier", "module", "répertoire"]
GOAL_KW      = ["but", "objectif", "quoi", "projet", "sert", "utilité", "rôle", "résumé", "présente"]

def retriever(state: RAGState) -> RAGState:
    query  = state.get("rewritten_query") or state["question"]
    vs     = get_vectorstore()

    # --- Filtre direct par nom de fichier ---
    filename = _extract_filename(query)
    if filename:
        total  = vs._collection.count()
        raw    = vs.get(limit=max(total, 1))
        matched = [
            Document(page_content=doc, metadata=meta)
            for doc, meta in zip(raw["documents"], raw["metadatas"])
            if filename in meta.get("source", "").lower()
        ]
        if matched:
            print(f"[Retriever] Filtre direct → {len(matched)} chunks pour '{filename}'")
            return {**state, "chunks": matched}

    # --- Recherche vectorielle avec reranking ---
    results = vs.similarity_search_with_score(query, k=RETRIEVAL_K)

    def score(item):
        doc, s = item
        src = doc.metadata.get("source", "").lower()
        bonus = 0
        if "readme" in src:  bonus += 0.4
        if "_structure" in src: bonus += 0.5
        if _is_type(query, STRUCTURE_KW) or _is_type(query, GOAL_KW):
            for kw, b in [("main",0.3),("config",0.25),("schema",0.2),
                          ("docker",0.2),("pyproject",0.2),("agent",0.15)]:
                if kw in src: bonus += b
        return s - bonus

    results.sort(key=score)
    top = [doc for doc, _ in results[:RETRIEVAL_TOP_N]]

    # Garantir _structure.md pour les questions de structure
    if _is_type(query, STRUCTURE_KW):
        present = {d.metadata.get("source","") for d in top}
        for doc, _ in results:
            if "_structure" in doc.metadata.get("source","").lower() and doc.metadata["source"] not in present:
                top.append(doc)
                break

    print(f"[Retriever] Vectorielle → {len(top)} chunks")
    return {**state, "chunks": top}


# ---------------------------------------------------------------------------
# Node 3 — Grader
# Évalue si les chunks récupérés sont suffisants pour répondre
# ---------------------------------------------------------------------------
def grader(state: RAGState) -> RAGState:
    question = state.get("rewritten_query") or state["question"]
    chunks   = state.get("chunks", [])
    retry    = state.get("retry_count", 0)

    if not chunks:
        return {**state, "grade": "retry", "retry_count": retry + 1}

    if retry >= 2:
        # On a déjà retried 2 fois, on génère quand même avec ce qu'on a
        return {**state, "grade": "relevant"}

    context_preview = "\n".join([c.page_content[:200] for c in chunks[:3]])

    prompt = f"""Tu es un évaluateur RAG. Dis si les extraits de code ci-dessous permettent de répondre à la question.
Réponds UNIQUEMENT par "oui" ou "non".

QUESTION: {question}

EXTRAITS:
{context_preview}

RÉPONSE (oui/non):"""

    verdict = call_llm(prompt, max_tokens=5, temperature=0).lower().strip()
    grade   = "relevant" if "oui" in verdict else "retry"
    print(f"[Grader] verdict='{verdict}' → grade='{grade}' (retry #{retry})")
    return {**state, "grade": grade, "retry_count": retry + 1}


# ---------------------------------------------------------------------------
# Node 4 — Retry Rewriter
# Si le grader dit "retry", reformule la query différemment pour un 2ème essai
# ---------------------------------------------------------------------------
def retry_rewriter(state: RAGState) -> RAGState:
    question = state.get("rewritten_query") or state["question"]

    prompt = f"""La recherche précédente n'a pas trouvé de résultats pertinents pour cette question.
Reformule-la de manière différente — utilise des synonymes ou une approche plus générale.
Retourne UNIQUEMENT la nouvelle question, sans explication.

QUESTION: {question}

NOUVELLE FORMULATION:"""

    new_query = call_llm(prompt, max_tokens=150, temperature=0.3)
    print(f"[RetryRewriter] nouvelle query: '{new_query}'")
    return {**state, "rewritten_query": new_query}


# ---------------------------------------------------------------------------
# Node 5 — Generator
# Génère la réponse finale avec le contexte et l'historique
# ---------------------------------------------------------------------------
def generator(state: RAGState) -> RAGState:
    question = state["question"]  # question originale pour la réponse
    query    = state.get("rewritten_query", question)
    chunks   = state.get("chunks", [])
    history  = state.get("history", [])

    context = "\n\n---\n\n".join([
        f"Fichier: {c.metadata.get('source','?')}\n{c.page_content}"
        for c in chunks
    ])

    history_text = ""
    if history:
        history_text = "\n".join([
            f"{'Utilisateur' if m['role'] == 'user' else 'Assistant'}: {m['content'][:300]}"
            for m in history[-8:]
        ])

    # Instruction contextuelle
    extra = ""
    if _is_type(query, STRUCTURE_KW):
        extra = "\n- La question porte sur la STRUCTURE. Liste précisément les dossiers/fichiers visibles. Ne génère pas de structure inventée."
    elif _is_type(query, GOAL_KW):
        extra = "\n- La question porte sur le BUT du projet. Déduis-le du README et du code. Sois direct en 3-5 phrases."

    prompt = f"""Tu es un expert en analyse de code et architecture logicielle.

Instructions :
- Réponds toujours en français, de manière claire et précise
- Base-toi UNIQUEMENT sur le contexte fourni
- Cite les fichiers sources quand c'est pertinent
- Si l'info n'est pas dans le contexte, dis-le clairement{extra}

{f"HISTORIQUE:{chr(10)}{history_text}{chr(10)}" if history_text else ""}
CONTEXTE:
{context}

QUESTION: {question}

RÉPONSE:"""

    answer  = call_llm(prompt, max_tokens=1500, temperature=0.1)
    sources = list({c.metadata.get("source","?") for c in chunks})
    return {**state, "answer": answer, "sources": sources}
