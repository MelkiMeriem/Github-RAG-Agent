import os
from dotenv import load_dotenv

load_dotenv()

# Paths
DB_PATH    = "./chroma_db"
REPO_PATH  = "./repo"
URL_FILE   = "./.repo_url"

# Models
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL  = "llama-3.3-70b-versatile"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Retrieval
RETRIEVAL_K        = 12
RETRIEVAL_TOP_N    = 8
MAX_RETRY_RETRIEVE = 2

# Chunking
CHUNK_SIZE         = 1000
CHUNK_OVERLAP      = 150
PRIORITY_CHUNK_SIZE = 4000

# Files
EXTENSIONS = {
    ".py", ".ipynb", ".js", ".ts", ".jsx", ".tsx",
    ".java", ".cpp", ".c", ".h", ".cs", ".go",
    ".rs", ".rb", ".php", ".swift", ".kt", ".scala",
    ".sh", ".bash", ".zsh", ".ps1",
    ".json", ".yaml", ".yml", ".toml", ".ini",
    ".xml", ".csv",
    ".md", ".rst", ".txt", ".tex",
    ".html", ".css", ".sql",
}

SKIP_DIRS = [
    "venv", ".git", "__pycache__", "node_modules",
    ".idea", ".vscode", "ragtest", "Scripts"
]

PRIORITY_FILES = {
    "readme", "requirements", "pyproject",
    "dockerfile", "docker-compose", "package.json", ".env"
}
