# Chhattisgarhi Farmer Agent — Prototype

Demo Streamlit app that answers farmer queries in Chhattisgarhi using local KB + optional Google Gemini.

How to run:
1. Deploy on https://share.streamlit.io by selecting this repository and `app.py`.
2. Add Gemini API key in Streamlit Secrets (`GEMINI_API_KEY`) or paste at runtime in the UI.
3. The app will load the `knowledge/` folder and build the retrieval index.

Notes:
- For a stable build, if FAISS fails on Streamlit Cloud switch to TF-IDF (edit requirements.txt).
- The app masks numeric pesticide dosages; do not use it as an authoritative pesticide source.
