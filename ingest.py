import os
import json
import git
import shutil
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

DB_PATH     = "./chroma_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

EXTENSIONS = {
    ".py", ".ipynb", ".js", ".ts", ".jsx", ".tsx",
    ".java", ".cpp", ".c", ".h", ".cs", ".go",
    ".rs", ".rb", ".php", ".swift", ".kt", ".scala",
    ".sh", ".bash", ".zsh", ".ps1",
    ".json", ".yaml", ".yml", ".toml", ".ini",
    ".xml", ".csv",
    ".md", ".rst", ".txt", ".tex",
    ".html", ".css",
    ".sql",
}

SKIP_DIRS = ["venv", ".git", "__pycache__", "node_modules", ".idea", ".vscode", "ragtest", "Scripts"]

def clone_repo(url: str, path: str) -> str:
    if os.path.exists(path):
        print(f"Repo déjà cloné dans {path}.")
        return path
    print(f"Clonage de {url}...")
    git.Repo.clone_from(url, path)
    print("Clonage terminé.")
    return path

def generate_tree(repo_path: str) -> Document:
    """
    Génère une représentation arborescente (style `tree`) de tous les fichiers
    du repo et la stocke comme un Document dédié dans le vectorstore.
    """
    root = Path(repo_path)
    lines = [root.name + "/"]

    def _walk(path: Path, prefix: str = ""):
        try:
            all_entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name))
        except PermissionError:
            return
        # Filtrer uniquement sur le nom direct de l'entrée, pas le chemin complet
        entries = [e for e in all_entries if e.name not in SKIP_DIRS]
        for i, entry in enumerate(entries):
            connector = "└── " if i == len(entries) - 1 else "├── "
            lines.append(prefix + connector + entry.name + ("/" if entry.is_dir() else ""))
            if entry.is_dir():
                extension = "    " if i == len(entries) - 1 else "│   "
                _walk(entry, prefix + extension)

    _walk(root)
    tree_text = "\n".join(lines)

    return Document(
        page_content=f"Structure complète du projet (arborescence) :\n\n```\n{tree_text}\n```",
        metadata={"source": "_structure.md", "type": ".md", "priority": "high"}
    )

def parse_notebook(filepath: Path, repo_path: str) -> list[Document]:
    docs = []
    try:
        nb = json.loads(filepath.read_text(encoding="utf-8", errors="ignore"))
        for i, cell in enumerate(nb.get("cells", [])):
            cell_type = cell.get("cell_type", "")
            if cell_type not in ("code", "markdown"):
                continue
            source = "".join(cell.get("source", []))
            if not source.strip():
                continue
            docs.append(Document(
                page_content=source,
                metadata={
                    "source": str(filepath.relative_to(repo_path)),
                    "type": ".ipynb",
                    "cell_type": cell_type,
                    "cell_index": i
                }
            ))
    except Exception as e:
        print(f"Erreur notebook {filepath}: {e}")
    return docs

def load_files(repo_path: str) -> list[Document]:
    docs = []
    skipped = 0

    for filepath in Path(repo_path).rglob("*"):
        if not filepath.is_file():
            continue
        if filepath.suffix not in EXTENSIONS:
            skipped += 1
            continue
        if any(p in filepath.parts for p in SKIP_DIRS):
            continue

        if filepath.suffix == ".ipynb":
            docs.extend(parse_notebook(filepath, repo_path))
            continue

        try:
            content = filepath.read_text(encoding="utf-8", errors="ignore")
            if not content.strip():
                continue
            rel_path = str(filepath.relative_to(repo_path))
            enriched = f"Fichier: {rel_path}\nType: {filepath.suffix}\n\n" + content
            docs.append(Document(
                page_content=enriched,
                metadata={
                    "source": rel_path,
                    "type": filepath.suffix
                }
            ))
        except Exception as e:
            print(f"Erreur lecture {filepath}: {e}")

    print(f"{len(docs)} fichiers/cellules chargés. ({skipped} fichiers ignorés)")
    return docs

def chunk_documents(docs: list[Document]) -> list[Document]:
    # Les documents à haute priorité (README, _structure) ne sont PAS chunckés
    priority_docs = [d for d in docs if d.metadata.get("priority") == "high"]
    normal_docs   = [d for d in docs if d.metadata.get("priority") != "high"]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=[
            "\n\n\n", "\n\n",
            "\nclass ", "\ndef ", "\nfunction ", "\nfunc ",
            "\nvoid ", "\npublic ", "\nprivate ", "\nstatic ",
            "\nconst ", "\nlet ", "\nvar ",
            "\n## ", "\n### ", "\n# ",
            "\n", " ", ""
        ]
    )
    chunks = splitter.split_documents(normal_docs)
    all_chunks = priority_docs + chunks  # priorité en premier
    print(f"{len(all_chunks)} chunks créés ({len(priority_docs)} docs prioritaires non chunckés).")
    return all_chunks

def build_vectorstore(chunks: list[Document]) -> Chroma:
    print("Génération des embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH
    )
    print("Vectorstore sauvegardé.")
    return vectorstore

def ingest(repo_url: str):
    local_path = "./repo"
    url_file   = "./.repo_url"

    if os.path.exists(url_file):
        with open(url_file) as f:
            old_url = f.read().strip()
        if old_url != repo_url:
            print("Nouveau repo détecté, suppression de l'ancien...")
            shutil.rmtree(local_path, ignore_errors=True)
            shutil.rmtree(DB_PATH, ignore_errors=True)

    with open(url_file, "w") as f:
        f.write(repo_url)

    clone_repo(repo_url, local_path)

    # 1. Générer l'arborescence en premier
    tree_doc = generate_tree(local_path)
    print("Arborescence générée.")

    # 2. Charger les fichiers normaux
    docs = load_files(local_path)

    # Marquer et enrichir les fichiers clés
    PRIORITY_FILES = {"readme", "requirements", "pyproject", "dockerfile", "docker-compose", "package.json", ".env"}
    for doc in docs:
        src = doc.metadata.get("source", "").lower()
        src_name = Path(src).name.lower()

        # Priorité haute pour les fichiers de config/meta
        if any(key in src_name for key in PRIORITY_FILES):
            doc.metadata["priority"] = "high"

        # Enrichissement sémantique : ajouter un en-tête descriptif
        # pour les fichiers dont le contenu est peu descriptif (liste de packages, etc.)
        if "requirements" in src_name:
            doc.page_content = (
                f"Dépendances et bibliothèques Python du projet (fichier {src}) :\n"
                f"Liste des packages requis, dépendances, librairies installées :\n\n"
                + doc.page_content
            )
        elif "dockerfile" in src_name:
            doc.page_content = (
                f"Configuration Docker, image de déploiement (fichier {src}) :\n\n"
                + doc.page_content
            )
        elif "docker-compose" in src_name:
            doc.page_content = (
                f"Configuration Docker Compose, services du projet (fichier {src}) :\n\n"
                + doc.page_content
            )
        elif "pyproject" in src_name or "package.json" in src_name:
            doc.page_content = (
                f"Configuration du projet, métadonnées, dépendances (fichier {src}) :\n\n"
                + doc.page_content
            )

    # 3. Injecter le tree doc dans la liste
    docs.insert(0, tree_doc)

    chunks = chunk_documents(docs)
    build_vectorstore(chunks)
    print("Ingestion terminée !")

if __name__ == "__main__":
    url = input("URL du repo GitHub : ").strip()
    ingest(url)
