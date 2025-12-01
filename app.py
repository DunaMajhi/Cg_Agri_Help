import os
import json
import glob
from pathlib import Path
from typing import List
import re

import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm

# Try to import google generative AI client
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except Exception:
    GENAI_AVAILABLE = False

# --- Config / secrets ---
DEFAULT_CONFIG = {
    "GEMINI_API_KEY": "",
    "GEMINI_MODEL": "gemini-2.0-flash",
    "TOP_K": 3,
    "REPLY_MAX_TOKENS": 256,
    "LANG_HINT": "chhattisgarhi",
    "FALLBACK_REPLY": "Ma nischit nahi haun — kripya nikat ke Krishi Vistar Kendra se sampark karo."
}

CONFIG_PATH = "config.json"
cfg = DEFAULT_CONFIG.copy()
if Path(CONFIG_PATH).exists():
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
            cfg.update(json.load(fh))
    except Exception:
        pass

# Streamlit secrets override (use st.secrets in Streamlit Cloud if desired)
if hasattr(st, 'secrets') and 'GEMINI_API_KEY' in st.secrets:
    cfg['GEMINI_API_KEY'] = st.secrets['GEMINI_API_KEY']

# UI header
st.set_page_config(page_title='Chhattisgarhi Farmer Agent', layout='wide')
st.title('Chhattisgarhi Farmer Agent — Prototype')
st.markdown('Upload short `.txt` advisories (one file = one advisory). Ask farmers\\' questions and get short Chhattisgarhi answers grounded in the KB.')

# Sidebar controls
st.sidebar.header('Settings')
top_k = st.sidebar.number_input('Top-k retrieval', value=int(cfg.get('TOP_K', 3)), min_value=1, max_value=10)
model_name = st.sidebar.text_input('Gemini model (if using)', value=cfg.get('GEMINI_MODEL', 'gemini-2.0-flash'))
max_tokens = st.sidebar.number_input('Reply max tokens', value=int(cfg.get('REPLY_MAX_TOKENS', 256)), min_value=50, max_value=2000)

# Knowledge upload
st.header('Knowledge Base')
uploaded = st.file_uploader('Upload one or more small .txt advisory files', accept_multiple_files=True, type=['txt'])
use_example = st.checkbox('Use built-in example KB (demo)', value=True)

KB_DIR = 'knowledge'
os.makedirs(KB_DIR, exist_ok=True)

if uploaded:
    for f in uploaded:
        out_path = Path(KB_DIR) / f.name
        with open(out_path, 'wb') as fh:
            fh.write(f.read())
    st.success(f'Saved {len(uploaded)} uploaded file(s) to `{KB_DIR}/`')

if use_example and not any(Path(KB_DIR).glob('*.txt')):
    st.info('Seeding example KB from preloaded files (if present)')

# Load docs
def load_docs(kb_dir: str) -> List[dict]:
    docs = []
    files = sorted(glob.glob(f"{kb_dir}/*.txt"))
    for p in files:
        with open(p, 'r', encoding='utf-8') as fh:
            t = fh.read().strip()
            if t:
                docs.append({'id': Path(p).name, 'text': t})
    return docs

docs = load_docs(KB_DIR)
st.write(f'Knowledge documents: {len(docs)}')
if len(docs) <= 0:
    st.warning('No knowledge files found. Upload .txt advisories or enable example KB.')
    st.stop()

# Embedding model (cache)
@st.cache_resource(show_spinner=False)
def get_embedder(name='sentence-transformers/all-MiniLM-L6-v2'):
    return SentenceTransformer(name)

embedder = get_embedder()

# Build FAISS index (cache)
@st.cache_resource(show_spinner=False)
def build_index(texts):
    embs = embedder.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    faiss.normalize_L2(embs)
    dim = embs.shape[1]
    idx = faiss.IndexFlatIP(dim)
    idx.add(embs)
    return idx, embs

texts = [d['text'] for d in docs]
index, doc_embs = build_index(texts)

# Retrieval helper
def retrieve(query: str, k: int = 3):
    q_emb = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, k)
    results = []
    for i in I[0]:
        if i < len(docs):
            results.append(docs[i])
    return results

# Build prompt for Gemini
def build_prompt(user_q: str, retrieved: List[dict]) -> str:
    refs = "\n\n".join([f"---\n{r['id']}\n{r['text']}" for r in retrieved]) if retrieved else ""
    prompt = f\"\"\"
You are an expert agricultural advisor for smallholder farmers. Use the reference passages to answer the farmer's question in simple Chhattisgarhi (use clear words). Keep answers short (1-3 sentences). If unsure, reply: \"{cfg.get('FALLBACK_REPLY')}\"

References:
{refs}

Farmer question:
{user_q}

Answer in Chhattisgarhi:
\"\"\"
    return prompt.strip()

# Call Gemini (sync)
def call_gemini_sync(prompt: str, model: str, max_tokens: int, api_key: str) -> str:
    if not GENAI_AVAILABLE or not api_key:
        return cfg.get('FALLBACK_REPLY')
    try:
        genai.configure(api_key=api_key)
        r = genai.generate_text(model=model, prompt=prompt, max_output_tokens=max_tokens)
        if hasattr(r, 'text') and r.text:
            return r.text.strip()
        if isinstance(r, dict) and 'candidates' in r and r['candidates']:
            return r['candidates'][0].get('content','').strip()
        return str(r)[:1000]
    except Exception as e:
        st.error(f'Gemini call failed: {e}')
        return cfg.get('FALLBACK_REPLY')

# Simple numeric dosage masker
def mask_numeric_dosages(text: str) -> str:
    masked = re.sub(r"\b\d{1,3}\s?(ml|g|kg|l|lit|लिटर|मिली|ग्राम|मि)\b", "[label पर देखें]", text, flags=re.IGNORECASE)
    masked = re.sub(r"\b\d{1,3}(\-\d{1,3})?\b", lambda m: m.group(0) if len(m.group(0))<2 else "[label पर देखें]", masked)
    if '[label पर देखें]' in masked:
        masked += "\n\nKripya pesticide ke liye hammesha label aur adhikarik Krishi Vistar Kendra dekhen."
    return masked

# Interaction UI
st.header('Ask a question')
col1, col2 = st.columns([3,1])
with col1:
    user_q = st.text_input("Type farmer's question (Chhattisgarhi / Hindi / English)", "")
with col2:
    ask_btn = st.button('Ask')

if ask_btn and user_q.strip():
    with st.spinner('Retrieving...'):
        retrieved = retrieve(user_q, k=top_k)
    st.subheader('Retrieved passages')
    for r in retrieved:
        st.markdown(f"**{r['id']}**")
        st.write(r['text'])

    prompt = build_prompt(user_q, retrieved)
    st.markdown('**Prompt sent to LLM (for debugging)**')
    st.code(prompt[:2000])

    gemini_key = cfg.get('GEMINI_API_KEY') or st.text_input('Paste Gemini API key (optional for this run)', type='password')
    if gemini_key:
        with st.spinner('Calling Gemini...'):
            reply = call_gemini_sync(prompt, model_name, max_tokens, gemini_key)
    else:
        reply = retrieved[0]['text'] if retrieved else cfg.get('FALLBACK_REPLY')

    safe_reply = mask_numeric_dosages(reply)
    st.subheader('Agent reply (Chhattisgarhi)')
    st.write(safe_reply)

    fb = st.radio('Was this helpful?', ('Yes','No'), index=0, key='fb')
    if st.button('Submit feedback'):
        log_line = {'q': user_q, 'reply': safe_reply, 'helpful': fb}
        with open('feedback.log','a',encoding='utf-8') as fh:
            fh.write(json.dumps(log_line, ensure_ascii=False) + '\\n')
        st.success('Thanks — feedback saved.')

st.markdown('---')
st.caption('Prototype: do not provide exact pesticide dosages. Always verify with KVK or extension officer.')
