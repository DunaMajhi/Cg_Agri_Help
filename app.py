import os
import json
import glob
from pathlib import Path
from typing import List
import re

import streamlit as st
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import tqdm

# Try Gemini
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except:
    GENAI_AVAILABLE = False

# ---- Load config ----
DEFAULT_CONFIG = {
    "GEMINI_API_KEY": "",
    "GEMINI_MODEL": "gemini-2.0-flash",
    "TOP_K": 3,
    "REPLY_MAX_TOKENS": 256,
    "LANG_HINT": "chhattisgarhi",
    "FALLBACK_REPLY": "Ma nischit nahi haun — kripya nikat ke Krishi Vistar Kendra se sampark karo."
}

cfg = DEFAULT_CONFIG.copy()
if Path("config.json").exists():
    try:
        with open("config.json", "r", encoding="utf-8") as f:
            cfg.update(json.load(f))
    except:
        pass

if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
    cfg["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]

# ---- UI ----
st.set_page_config(page_title="Chhattisgarhi Farmer Agent", layout="wide")
st.title("Chhattisgarhi Farmer Agent — Prototype (TF-IDF Version)")

st.header("Knowledge Base")
uploaded = st.file_uploader("Upload .txt files", accept_multiple_files=True)
use_example = st.checkbox("Use built-in example KB", value=True)

KB_DIR = "knowledge"
os.makedirs(KB_DIR, exist_ok=True)

if uploaded:
    for f in uploaded:
        with open(os.path.join(KB_DIR, f.name), "wb") as out:
            out.write(f.read())
    st.success("Uploaded!")

if use_example and not any(Path(KB_DIR).glob("*.txt")):
    demo = {
        "paddy_irrigation.txt": "Dhan ma adhik paani mat rakho. 5-7 din ma ek baar paani uchit hae.",
        "paddy_bph_signs.txt": "BPH nishan: patta pe pilingan, paudha kamzor. Photo bhejo.",
        "storage_tip.txt": "Anaj dry jagah ma rakho, humidity se bacha ke rakho."
    }
    for name, text in demo.items():
        with open(f"{KB_DIR}/{name}", "w", encoding="utf-8") as f:
            f.write(text)

# ---- Load docs ----
docs = []
paths = sorted(glob.glob(f"{KB_DIR}/*.txt"))
for p in paths:
    with open(p, "r", encoding="utf-8") as f:
        txt = f.read().strip()
        if txt:
            docs.append({"id": Path(p).name, "text": txt})

if not docs:
    st.warning("No KB files found.")
    st.stop()

texts = [d["text"] for d in docs]

# ---- Build TF-IDF ----
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

def retrieve(q, k=3):
    v = vectorizer.transform([q])
    sims = (X @ v.T).toarray().squeeze()
    idx = sims.argsort()[::-1][:k]
    return [docs[i] for i in idx]

# ---- Gemini Prompt ----
def make_prompt(q, refs):
    ref_text = "\n\n".join([f"---\n{r['id']}\n{r['text']}" for r in refs])
    return f"""
You are an agriculture assistant. Answer briefly in simple Chhattisgarhi.
If unsure, say: "{cfg['FALLBACK_REPLY']}".

References:
{ref_text}

Farmer question:
{q}

Answer:
""".strip()

def call_gemini(prompt):
    key = cfg.get("GEMINI_API_KEY", "")
    if not GENAI_AVAILABLE or not key:
        return cfg["FALLBACK_REPLY"]

    genai.configure(api_key=key)
    try:
        r = genai.generate_text(
            model=cfg["GEMINI_MODEL"],
            prompt=prompt,
            max_output_tokens=cfg["REPLY_MAX_TOKENS"]
        )
        if hasattr(r, "text"):
            return r.text.strip()
        return cfg["FALLBACK_REPLY"]
    except:
        return cfg["FALLBACK_REPLY"]

# ---- Input UI ----
st.header("Ask a question")
user_q = st.text_input("Type question (Chhattisgarhi/Hindi/English)")
if st.button("Ask") and user_q.strip():
    hits = retrieve(user_q, k=cfg["TOP_K"])
    st.subheader("Retrieved KB")
    for h in hits:
        st.markdown(f"**{h['id']}**")
        st.write(h["text"])

    prompt = make_prompt(user_q, hits)
    st.markdown("### Prompt (debug)")
    st.code(prompt)

    reply = call_gemini(prompt)
    st.markdown("### Reply")
    st.write(reply)
