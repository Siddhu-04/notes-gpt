# 📖 notes-gpt

Chat with your own PDFs — upload your notes, ask questions, get cited answers.

- Hybrid retrieval: BM25 + semantic search (BAAI/bge-small-en-v1.5), reranked with BAAI/bge-reranker-base
- Answers cite `[filename p.N]` so you can verify every claim
- Each user's uploads are session-only, in-memory, and never shared with other visitors
- Powered by Groq (Llama 3.3 70B) for fast streaming responses

## Run locally
\`\`\`bash
uv add streamlit pymupdf chromadb sentence-transformers rank-bm25 openai python-dotenv
echo "GROQ_API_KEY=your_key" > .env
cd notes_gpt
uv run streamlit run app.py
\`\`\`

## Live demo
https://notes-gpt.streamlit.app/

Refer above link for trying it by yourself...🙃🤞🏻

## NOTE
"notes-gpt" is just a Retrieval-Augmented Generation (RAG) project I built as an experiment to understand the full RAG pipeline hands-on, not just wiring together a framework, but implementing chunking, hybrid retrieval, reranking, and citation-grounded generation from scratch.
Each user's uploaded PDFs are indexed into an isolated, in-memory vector store scoped to their browser session — nothing is persisted or shared between users. Answers are generated only from retrieved context, with the model instructed to say "I don't know" rather than hallucinate when nothing relevant is found. Hope you try it...
