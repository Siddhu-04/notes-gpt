# RAG Evaluation — notes-gpt

Systematic evaluation of the retrieval and generation quality of notes-gpt, using [RAGAS](https://github.com/explodinggpt/ragas) on a 20-question hand-curated eval set covering all 20 sample PDFs.

## Results

| Metric | Score | What it measures |
|---|---|---|
| Faithfulness | 0.95 | Are answers grounded in retrieved context (no hallucination)? |
| Answer Relevancy | 0.88 | Do answers actually address the question asked? |
| Context Precision | 0.98 | Did retrieval find the right chunks? |

All three scores are above the 0.85 threshold generally considered production-ready.

## Methodology

- **Answer generation:** LlamaIndex-based retrieval pipeline (`llamaindex_version.py`) using `BAAI/bge-small-en-v1.5` embeddings and Groq's `llama-3.3-70b-versatile`.
- **Judge model:** Groq's `llama-3.1-8b-instant`, deliberately different from the answer-generation model to avoid self-evaluation bias.
- **Embeddings for scoring:** `BAAI/bge-small-en-v1.5` (same model used in the actual retrieval pipeline), run locally — no OpenAI dependency.

## Files

- `eval_questions.json` — 20 hand-written question/ground-truth pairs, one per source PDF
- `run_eval.py` — runs each question through the RAG pipeline, capturing answers + retrieved contexts
- `eval_results.json` — output of the above (answers + contexts per question)
- `score.py` — runs RAGAS scoring against the results
- `scores.json` — final per-question metric breakdown

## Reproduce

\`\`\`bash
uv run python evals/llamaindex_version.py   # sanity check pipeline
uv run python evals/run_eval.py             # generate answers for all 20 questions
uv run python evals/score.py                # score with RAGAS
\`\`\`
