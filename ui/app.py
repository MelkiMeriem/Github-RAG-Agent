import streamlit as st
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from ingest import ingest
from rag.graph import ask

st.set_page_config(page_title="GitHub RAG", page_icon="🔍")
st.title("GitHub Repo Q&A")
st.caption("Pose des questions en langage naturel sur n'importe quel repo GitHub.")

with st.sidebar:
    st.header("Repo GitHub")
    repo_url  = st.text_input("URL du repo", placeholder="https://github.com/user/repo")
    index_btn = st.button("Indexer le repo", type="primary")

    if index_btn and repo_url:
        with st.spinner("Clonage et indexation en cours..."):
            try:
                from core.vectorstore import reset_vectorstore
                ingest(repo_url)
                reset_vectorstore()  # reset le singleton après nouvelle ingestion
                st.session_state.repo_indexed = True
                st.session_state.repo_url     = repo_url
                st.session_state.history      = []
                st.success("Repo indexé avec succès !")
            except Exception as e:
                st.error(f"Erreur : {e}")

    if st.session_state.get("repo_url"):
        st.caption(f"Repo actuel :\n`{st.session_state.repo_url}`")

    if st.session_state.get("repo_indexed"):
        if st.button("🗑️ Effacer la conversation"):
            st.session_state.history = []
            st.rerun()

if "history" not in st.session_state:
    st.session_state.history = []

if not st.session_state.get("repo_indexed"):
    st.info("Entre l'URL d'un repo GitHub dans la sidebar et clique sur **Indexer le repo** pour commencer.")
else:
    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Pose une question sur le repo...")

    if question:
        st.session_state.history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Analyse en cours..."):
                result = ask(question, history=st.session_state.history[:-1])
            st.markdown(result["answer"])
            with st.expander("Sources utilisées"):
                for src in result["sources"]:
                    st.code(src)

        st.session_state.history.append({"role": "assistant", "content": result["answer"]})
