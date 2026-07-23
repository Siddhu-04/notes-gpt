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
Refer above link for trying it by yourself...Hope it works...🥲🤞🏻
