# GitHub RAG — Agentic Q&A

**Un système de question-réponse alimenté par l'IA pour vos repos GitHub.**

Pose des questions en langage naturel sur **n'importe quel repo GitHub** et reçois des réponses précises et contextualisées. Le système clone automatiquement le repo, indexe tous les fichiers, et utilise un pipeline RAG (Retrieval-Augmented Generation) avec un agent intelligent pour naviguer dans la base de code.

## Fonctionnalités

- **Indexation automatique** : Clone et indexe tous les fichiers du repo (Python, JavaScript, Java, etc.)
- **Recherche intelligente** : Combine recherche vectorielle (embeddings) et recherche directe par nom de fichier
- **Réranking automatique** : Évalue la pertinence des chunks récupérés
- **Agentic RAG** : Retry automatique avec reformulation de la question si les premiers résultats ne sont pas pertinents
- **Historique conversationnel** : Résout les coréférences (pronoms, références implicites) grâce à l'historique
- **Interface web** : Interface Streamlit moderne et intuitive
- **Support multi-formats** : Code, markdown, notebooks, JSON, YAML, SQL, etc.

## Structure du projet

```
github-rag/
├── core/                   # Composants partagés et configurables
│   ├── config.py           # Configuration centralisée (paths, modèles, paramètres)
│   ├── llm.py              # Client Groq singleton + wrapper call_llm()
│   └── vectorstore.py      # ChromaDB singleton pour accès persistent
│
├── rag/                    # Pipeline Agentic RAG (basé sur LangGraph)
│   ├── state.py            # TypedDict RAGState - état du graphe
│   ├── nodes.py            # 5 nodes : rewriter, retriever, grader, retry, generator
│   └── graph.py            # Graphe LangGraph orchestré + fonction ask() publique
│
├── ui/
│   └── app.py              # Interface Streamlit (chat, indexation, gestion historique)
│
├── ingest.py               # Script d'ingestion : clone repo + extraction + chunking
├── requirements.txt        # Dépendances Python
├── .env                    # Clés API (GROQ_API_KEY)
└── README.md               # Ce fichier
```

## Architecture du pipeline RAG

```
Entrée utilisateur (question)
    ↓
┌─────────────────────────────────────────┐
│ [query_rewriter]                        │
│ Reformule la question en utilisant      │
│ l'historique conversationnel             │
│ (résout coréférences, pronoms, etc.)    │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ [retriever]                             │
│ • Recherche directe (noms de fichiers)  │
│ • Recherche vectorielle (embeddings)    │
│ • Reranking (contextual relevance)      │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ [grader]                                │
│ Évalue : les chunks sont-ils pertinents │
│ pour répondre à la question ?            │
└─────────────────────────────────────────┘
    ↓
    ├─ Pertinent
    │   ↓
    │   [generator] → Génère la réponse
    │   ↓
    │   Réponse finale + sources
    │
    └─ Non pertinent (max 2 tentatives)
        ↓
        [retry_rewriter] → Reformule différemment
        ↓
        [retriever] → Nouvelle recherche
        ↓
        (boucle jusqu'à succès ou limite atteinte)
```

## Installation et lancement

### Prérequis
- Python 3.10+
- Clé API Groq (gratuite sur [console.groq.com](https://console.groq.com))
- Git

### Étapes

1. **Cloner le repo**
   ```bash
   git clone https://github.com/yourusername/github-rag.git
   cd github-rag
   ```

2. **Créer un environnement virtuel**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Sur Windows: venv\Scripts\activate
   ```

3. **Installer les dépendances**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configurer les variables d'environnement**
   ```bash
   cp .env.example .env
   ```
   Puis éditer `.env` et ajouter votre clé Groq :
   ```
   GROQ_API_KEY=your_api_key_here
   ```

5. **Lancer l'interface**
   ```bash
   streamlit run ui/app.py
   ```
   
   L'interface s'ouvrira automatiquement à `http://localhost:8501`

## Utilisation

### Interface utilisateur

1. **Sidebar gauche** : Entrez l'URL d'un repo GitHub (ex: `https://github.com/langchain-ai/langchain`)
2. **Bouton "Indexer le repo"** : Lance le clonage et l'indexation (peut prendre quelques minutes selon la taille du repo)
3. **Zone principale** : Posez vos questions en langage naturel
4. **Expander "Sources utilisées"** : Consultez les fichiers et chunks utilisés pour générer la réponse

### Exemples de questions

```
# Sur un repo LangChain
- "Comment créer une chaîne LangChain simple ?"
- "Montre-moi un exemple d'utilisation de retrievers"
- "Quels sont les modèles LLM supportés ?"

# Sur un repo React
- "Comment utiliser les hooks ?"
- "Explique le contexte React"
- "Montre un exemple de composant fonctionnel"
```

## Composants clés

### `core/config.py`
Centralise toute la configuration du projet :
- Paths (DB, repos)
- Modèles d'embedding et LLM
- Paramètres de chunking, recherche, etc.

### `core/llm.py`
Wrapper singleton autour du client Groq :
- Réutilise la même connexion
- Logs et gestion d'erreurs uniformes
- Fonction `call_llm()` publique

### `core/vectorstore.py`
Singleton ChromaDB :
- Chargement lazy de la base vectorielle
- Reset possible après réindexation

### `ingest.py`
Pipeline d'ingestion complet :
- Clonage du repo via GitPython
- Extraction des fichiers supportés
- Parsing des Jupyter notebooks
- Génération d'une arborescence du projet
- Chunking intelligent avec séparateurs personnalisés
- Génération des embeddings via HuggingFace

### `rag/state.py`
État du graphe LangGraph (TypedDict) :
- Question utilisateur
- Historique conversationnel
- Documents récupérés
- Pertinence des documents
- Nombre de tentatives de retry

### `rag/nodes.py`
Les 5 nœuds du pipeline :

1. **query_rewriter** : Reformule la question avec contexte historique
2. **retriever** : Récupère les chunks pertinents (recherche combinée)
3. **grader** : Évalue la pertinence des résultats
4. **generator** : Génère la réponse finale
5. **retry_rewriter** : Reformule en cas de pertinence insuffisante

### `rag/graph.py`
Orchestration LangGraph + API publique `ask(question, history)`

### `ui/app.py`
Interface Streamlit :
- Gestion des sessions utilisateur
- Upload de repos
- Historique conversationnel
- Affichage des sources

## Formats de fichiers supportés

| Catégorie | Extensions |
|-----------|-----------|
| Code | `.py`, `.js`, `.ts`, `.jsx`, `.tsx`, `.java`, `.cpp`, `.c`, `.h`, `.cs`, `.go`, `.rs`, `.rb`, `.php`, `.swift`, `.kt`, `.scala` |
| Config | `.json`, `.yaml`, `.yml`, `.toml`, `.ini`, `.xml` |
| Docs | `.md`, `.rst`, `.txt`, `.tex` |
| Web | `.html`, `.css` |
| Data | `.csv`, `.sql` |
| Notebooks | `.ipynb` |

## Sécurité

- Clés API stockées dans `.env` (jamais commitées)
- ChromaDB persiste localement
- Repos clonés dans `./repo/` (isolé)
- Validation des URLs GitHub

## Ressources

- [LangGraph Documentation](https://python.langchain.com/docs/langgraph)
- [ChromaDB Documentation](https://docs.trychroma.com)
- [Groq API Documentation](https://console.groq.com/docs)
- [Streamlit Documentation](https://docs.streamlit.io)

