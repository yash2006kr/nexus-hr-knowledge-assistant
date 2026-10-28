# NEXUS HR Knowledge Assistant

A Streamlit demonstration app for searching HR policy documents with semantic retrieval. It chunks policy text, creates embeddings with `sentence-transformers/all-MiniLM-L6-v2`, and uses FAISS for similarity search. The interface includes a chat-style search page, a knowledge base for PDF, DOCX, and TXT uploads, and retrieval settings.

> **Course project:** This was a task given as part of the Generative AI (GenAI) course by Samsung Innovation Campus at my college.

## Features

- Semantic search over policy passages using sentence-transformer embeddings and FAISS.
- Upload and index PDF, DOCX, and TXT documents for the current Streamlit session.
- Built-in illustrative HR policies for trying the demo.
- Configurable result count and similarity threshold.

## Run locally

Requires Python 3.10 or newer.

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit. The first run downloads the embedding model.

## Files

- `app.py` — Streamlit application.
- `Copy_of_RAG_Demo.ipynb` — cleaned educational notebook; saved cell outputs were removed.

## Notes and limitations

- Retrieved passages should be checked against the original policy document.
- The built-in policies are demonstration content, not real company policies.
- Scanned PDFs need OCR before their text can be extracted.
- Uploaded documents are kept in Streamlit session memory and are not stored in a persistent database.
- This project performs retrieval and displays source passages; it does not use an LLM to generate a synthesized answer.
- The notebook includes an optional ngrok section for temporary demos. Do not commit ngrok tokens or other credentials.
