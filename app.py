import os
import re
import json
import datetime
import tempfile
import shutil
from typing import Optional, Dict, Any

import streamlit as st
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# LangChain & Ollama imports
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader, TextLoader, Docx2txtLoader
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

# ─── CONFIGURATION ──────────────────────────────────────────────────────────────

# Ollama server URL (set via environment variable or fallback to localhost)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Campaign categories and regex patterns
CAMPAIGN_PATTERNS = {
    "Digital Marketing":      [r"digital marketing", r"online marketing", r"website", r"app"],
    "Content Marketing":      [r"content marketing", r"blog", r"article", r"whitepaper"],
    "Social Media Marketing": [r"social media", r"facebook", r"instagram", r"twitter", r"linkedin"],
    "Email Marketing":        [r"email", r"newsletter", r"mail campaign"],
    "SEO & SEM":              [r"seo", r"search engine", r"sem", r"google ads", r"ppc"],
    "Influencer Marketing":   [r"influencer", r"celebrity endorsement", r"collaborat"],
    "Brand Development":      [r"brand", r"identity", r"positioning"],
    "Market Research":        [r"market research", r"survey", r"analysis", r"trend"]
}
MARKETING_CATEGORIES = list(CAMPAIGN_PATTERNS.keys())
CATEGORY_DESCRIPTIONS = {
    "Digital Marketing":      "Strategies for online marketing channels including websites, apps, social media, email, and search engines.",
    "Content Marketing":      "Creating and distributing valuable content to attract and engage a target audience.",
    "Social Media Marketing": "Strategies specific to social platforms like Instagram, Facebook, LinkedIn, Twitter, and TikTok.",
    "Email Marketing":        "Direct marketing strategies using email to promote products or services.",
    "SEO & SEM":              "Techniques to improve search engine visibility and paid search strategies.",
    "Influencer Marketing":   "Partnering with influential people to increase brand awareness or drive sales.",
    "Brand Development":      "Strategies to build, strengthen and promote a company's brand identity.",
    "Market Research":        "Methods to gather and analyze information about consumers, competitors, and market trends."
}

# AIDA model description
AIDA_DESCRIPTION = (
    "**AIDA Marketing Model**\n"
    "1. **Attention**: Capture the audience's awareness with compelling hooks or visuals.\n"
    "2. **Interest**: Maintain curiosity by highlighting benefits and features.\n"
    "3. **Desire**: Build emotional engagement through value propositions and social proof.\n"
    "4. **Action**: Prompt a clear next step such as purchase, sign-up, or inquiry.\n\n"
    "Use this framework to guide prospects from awareness to conversion."
)

# ─── HELPER FUNCTIONS ────────────────────────────────────────────────────────────

def classify_campaign(text: str) -> Optional[str]:
    text_lower = text.lower()
    for campaign, patterns in CAMPAIGN_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, text_lower):
                return campaign
    return None

@st.cache_data(show_spinner=False)
def load_lottieurl(url: str) -> Optional[Dict[str, Any]]:
    try:
        r = requests.get(url)
        return r.json() if r.status_code == 200 else None
    except:
        return None

def check_ollama_connection() -> (bool, str):
    try:
        client = ChatOllama(base_url=OLLAMA_BASE_URL, model="llama2", temperature=0.3)
        r = client.invoke("ping")
        return True, getattr(r, "content", str(r))
    except Exception as e:
        return False, str(e)

@st.cache_resource(show_spinner=False)
def get_ollama_client(model: str = "llama2", temp: float = 0.3):
    return ChatOllama(base_url=OLLAMA_BASE_URL, model=model, temperature=temp)

@st.cache_resource(show_spinner=False)
def process_documents(docs_list):
    with tempfile.TemporaryDirectory() as td:
        paths = []
        for file in docs_list:
            p = os.path.join(td, file.name)
            with open(p, "wb") as f:
                f.write(file.getbuffer())
            paths.append(p)

        texts = []
        for p in paths:
            try:
                if p.endswith(".pdf"):
                    loader = PDFPlumberLoader(p)
                elif p.endswith(".docx"):
                    loader = Docx2txtLoader(p)
                else:
                    loader = TextLoader(p)
                texts.extend(loader.load())
            except Exception as e:
                st.error(f"Load error {os.path.basename(p)}: {e}")

        if not texts:
            st.error("No documents loaded.")
            return None

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(texts)
        if not chunks:
            st.error("No chunks generated.")
            return None

        dbdir = "./marketing_db"
        if os.path.exists(dbdir):
            shutil.rmtree(dbdir, onerror=lambda fn, path, exc: (os.chmod(path, 0o777), fn(path)))

        for emb_model in ["nomic-embed-text", "all-MiniLM", "llama2"]:
            try:
                st.info(f"Embedding with {emb_model}…")
                emb = OllamaEmbeddings(base_url=OLLAMA_BASE_URL, model=emb_model)
                if not emb.embed_query(chunks[0].page_content[:30]):
                    st.warning(f"{emb_model} produced an empty embedding")
                    continue
                store = Chroma.from_documents(chunks, emb, persist_directory=dbdir)
                st.success(f"Vectors stored with {emb_model}")
                return store
            except Exception as e:
                st.warning(f"{emb_model} failed: {e}")

        st.error("All embedding models failed.")
        return None

def get_retriever():
    vs = st.session_state.get("vector_store")
    return vs.as_retriever(search_type="mmr", search_kwargs={"k": 6, "fetch_k": 8}) if vs else None

def get_prompt():
    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are a marketing QA system. Use only provided docs and structure answers with AIDA."
        ),
        HumanMessagePromptTemplate.from_TEMPLATE(
            "Context:\n{context}\n\nQuestion: {question}"
        ),
    ])

def init_qa_chain():
    if not st.session_state.get("qa_chain") and st.session_state.get("vector_store"):
        try:
            llm = get_ollama_client("llama2", 0.1)
            retr = get_retriever()
            if retr:
                qa = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retr,
                    chain_type_kwargs={"prompt": get_prompt()}
                )
                st.session_state.qa_chain = qa
                return qa
        except Exception as e:
            st.error(f"QA init error: {e}")
    return st.session_state.get("qa_chain")

def format_resp(r):
    return getattr(r, "content", str(r)).strip()

def save_history(msgs, fname=None):
    os.makedirs("chat_histories", exist_ok=True)
    if not fname:
        fname = f"chat_{datetime.datetime.now():%Y%m%d_%H%M%S}.json"
    path = os.path.join("chat_histories", fname)
    try:
        with open(path, "w") as f:
            json.dump(msgs, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Save error: {e}")
        return False

def load_history(fname):
    try:
        with open(os.path.join("chat_histories", fname)) as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Load error: {e}")
        return None

def init_state():
    ss = st.session_state
    ss.setdefault("messages", [])
    ss.setdefault("vector_store", None)
    ss.setdefault("qa_chain", None)
    ss.setdefault("llm", get_ollama_client("llama2", 0.3))
    ss.setdefault("selected_category", MARKETING_CATEGORIES[0])
    ss.setdefault("chat_started", False)
    ss.setdefault(
        "available_histories",
        [f for f in os.listdir("chat_histories") if f.endswith(".json")]
        if os.path.exists("chat_histories") else []
    )

def main():
    st.set_page_config(page_title="Marketing Advisor", page_icon="📊", layout="wide")
    init_state()

    with st.sidebar:
        st.title("Marketing Advisor")

        cat = st.selectbox(
            "Focus area",
            MARKETING_CATEGORIES,
            index=MARKETING_CATEGORIES.index(st.session_state.selected_category)
        )
        if cat != st.session_state.selected_category:
            st.session_state.selected_category = cat
        st.info(CATEGORY_DESCRIPTIONS[cat])
        st.markdown("---")

        if st.button("Check Ollama Connection"):...
