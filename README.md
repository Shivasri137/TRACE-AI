# TRACE AI — Evidence-Based Claim Verification

**Don't just generate answers. Verify them.**

TRACE AI is an AI/ML MVP that evaluates claims against user-provided research excerpts or PDFs. It combines TF-IDF retrieval with an OpenRouter-hosted LLM to generate evidence-grounded verdicts and validate exact quotations.

## Features
- PDF text extraction with PyMuPDF (text-based PDFs; no OCR)
- Overlapping document chunks
- TF-IDF + cosine similarity retrieval
- OpenRouter API integration (OpenAI-compatible client)
- Verdicts: `SUPPORTED`, `CONTRADICTED`, `INSUFFICIENT_EVIDENCE`
- Source-ID and exact-quotation validation
- Streamlit dashboard and JSON report export
- Unit tests and GitHub Actions CI

## Architecture
1. Ingest pasted text or extract text from a PDF.
2. Split the source into overlapping chunks.
3. Rank passages against each claim using TF-IDF and cosine similarity.
4. Ask the configured OpenRouter model to classify claims only against retrieved evidence.
5. Validate verdict labels, source IDs, and exact quotation matches.
6. Display verdicts, evidence passages, and a downloadable JSON report.

## Run locally
Python 3.10+ recommended.

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Configure Streamlit secrets in `.streamlit/secrets.toml`:

```toml
OPENROUTER_API_KEY = "YOUR_KEY"
OPENROUTER_MODEL = "openrouter/free"
```

The OpenRouter free model router selects from currently available free models; availability and rate limits may change. You can set `OPENROUTER_MODEL` to another model available to your account. Never commit secrets.

## Tests
```bash
python -m pytest -v
```
The tests cover preprocessing and retrieval. They do not prove factual-verification accuracy or test a live API call.

## Deploy on Streamlit Community Cloud
1. Open https://share.streamlit.io/ and connect GitHub.
2. Select this repository, branch `main`, and main file `app.py`.
3. Add `OPENROUTER_API_KEY` and optionally `OPENROUTER_MODEL` under app Secrets.
4. Deploy, then test with a small source and a few claims.

## Limitations
- Checks only evidence supplied by the user; it does not browse the internet.
- TF-IDF retrieval is lexical and may miss paraphrases.
- Model verdicts can be wrong; exact quotation validation does not prove semantic support.
- Scanned PDFs need OCR and are not supported directly.
- Sentence splitting is intentionally simple.
- Retrieval relevance is not a truth score or calibrated confidence.
- OpenRouter model availability, cost, and rate limits depend on the selected model and account.

## Evaluation roadmap
Create a manually labeled set covering all three classes. Report accuracy, macro-F1, per-class precision/recall, retrieval recall, latency, and citation validity. Publish only measured results.

**Status:** MVP. Run tests and verify the configured OpenRouter model before relying on production behavior.
