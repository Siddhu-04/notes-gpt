import uuid
import streamlit as st
import chromadb
from ingest import load_pdf_bytes, chunk_pages, embed_chunks, index_to_chroma
from retrieve import get_embedder
from chat import answer_streaming

def setup_page():
    st.set_page_config(page_title="notes-gpt", page_icon="📖")
    st.title("notes-gpt---chat with your own PDFs")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "collection" not in st.session_state:
        st.session_state.collection = None

    with st.sidebar:
        st.header("Your documents...")
        st.caption("Uploads are kept in memory for this session only and are not shared with other users.")
        files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
        if files and st.button("Index documents"):
            with st.spinner("Indexing..."):
                client = chromadb.EphemeralClient()
                all_chunks = []
                for f in files:
                    pages = load_pdf_bytes(f.read(), f.name)
                    all_chunks.extend(chunk_pages(pages))
                embed_chunks(all_chunks, get_embedder())
                unique_name = f"session_{uuid.uuid4().hex}"
                st.session_state.collection = index_to_chroma(
                    all_chunks, collection_name=unique_name, client=client
                )
            st.success(f"Indexed {len(all_chunks)} chunks from {len(files)} PDFs")

        if st.session_state.collection is not None:
            count = st.session_state.collection.count()
            st.info(f"{count} chunks currently indexed for this session.")
            if st.button("Clear documents"):
                st.session_state.collection = None
                st.session_state.messages = []
                st.rerun()

def render_history():
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

def handle_chat():
    if q := st.chat_input("Ask anything from your uploaded PDFs..."):
        st.session_state.messages.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            if st.session_state.collection is None:
                ans = "Please upload and index at least one PDF in the sidebar first."
                st.markdown(ans)
            else:
                ans = st.write_stream(answer_streaming(q, st.session_state.collection))
        st.session_state.messages.append({"role": "assistant", "content": ans})

setup_page()
render_history()
handle_chat()
