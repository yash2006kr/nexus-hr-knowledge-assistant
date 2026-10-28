# Day 1 — NEXUS HR Knowledge Assistant

A Streamlit learning demo for semantic search over HR policy documents, created as a Samsung Innovation Campus GenAI course task.

## Run locally

Requires Python 3.10 or newer.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The first run downloads the embedding model. Uploaded files are held in session memory. Built-in policy text is illustrative, retrieved passages should be checked against source documents, and the app does not synthesize answers with an LLM.

The notebook is a cleaned copy of the original course demo with saved outputs removed.
