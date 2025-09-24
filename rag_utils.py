import os
import re
import io
from typing import List, Tuple

from dataclasses import dataclass
from pypdf import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.docstore.document import Document

from dotenv import load_dotenv
load_dotenv()

# ---- Google Drive (Service Account) ----
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-004")

# ================= Ingestão & Chunking =================
def pdfs_to_documents(pdf_paths: List[str]) -> List[Document]:
    docs = []
    for path in pdf_paths:
        try:
            reader = PdfReader(path)
        except Exception:
            # pypdf pode falhar em alguns PDFs; pule silenciosamente
            continue
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            meta = {"source": os.path.basename(path), "page": i+1, "path": path}
            if text.strip():
                docs.append(Document(page_content=text, metadata=meta))
    return docs

def chunk_documents(docs: List[Document], chunk_size=1200, chunk_overlap=160) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = []
    for d in docs:
        for c in splitter.split_text(d.page_content):
            meta = d.metadata.copy()
            chunks.append(Document(page_content=c, metadata=meta))
    return chunks

def build_vectorstore(chunks: List[Document], index_dir: str) -> FAISS:
    embedding = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)
    vs = FAISS.from_documents(chunks, embedding)
    os.makedirs(index_dir, exist_ok=True)
    vs.save_local(index_dir)
    return vs

def load_vectorstore(index_dir: str) -> FAISS:
    embedding = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)
    return FAISS.load_local(index_dir, embeddings=embedding, allow_dangerous_deserialization=True)

# ================= Citações =================
def render_citations(docs: List[Document]) -> str:
    uniq = []
    seen = set()
    for d in docs:
        tag = f"{d.metadata.get('source','?')} p.{d.metadata.get('page','?')}"
        if tag not in seen:
            seen.add(tag)
            uniq.append(tag)
    return "" if not uniq else "\n\n**Fontes consultadas:** " + "; ".join(uniq)

# ================= Heurística de cálculo =================
MATH_HINTS = [
    r"\bt[- ]?test\b", r"\bchi[- ]?square\b|\bqui[- ]?quadrado\b",
    r"\bANOVA\b", r"\bp[- ]?value\b|\bp-?valor\b", r"\bIC\b|\bintervalo de confiança\b",
    r"\bN\(\s*0\s*,\s*1\s*\)\b", r"\bdistribuiç(?:ão|ao)\b", r"\btabela t\b", r"\bz[- ]?score\b",
]
def might_need_calc(q: str) -> bool:
    ql = q.lower()
    if re.search(r"[0-9]", ql) and any(re.search(p, ql) for p in MATH_HINTS):
        return True
    return False

# ================= Execução segura de cálculo =================
def extract_python_block(text: str) -> str | None:
    m = re.search(r"```python(.*?)```", text, flags=re.S)
    return m.group(1).strip() if m else None

def run_calc_block(code: str) -> Tuple[bool, str]:
    import io, contextlib, re as _re
    import numpy as np
    from scipy import stats
    import math
    import sympy as sp
    allowed = {
        "np": np, "stats": stats, "math": math, "sp": sp,
        "t_ppf": stats.t.ppf, "t_cdf": stats.t.cdf,
        "norm_ppf": stats.norm.ppf, "norm_cdf": stats.norm.cdf,
        "chi2_ppf": stats.chi2.ppf, "chi2_cdf": stats.chi2.cdf,
        "f_ppf": stats.f.ppf, "f_cdf": stats.f.cdf,
    }
    if _re.search(r"\b(import|open|__|exec|eval|subprocess|os\.|sys\.)", code):
        return False, "Código bloqueado por segurança."
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, {"__builtins__": {}}, allowed)
        out = buf.getvalue().strip()
        return True, out if out else "(cálculo executado sem stdout)"
    except Exception as e:
        return False, f"Erro ao executar: {e}"

# ================= Prompt =================
SYSTEM_GUARDRAILS = """Você é um tutor de Inferência Estatística.
Responda APENAS com base nos trechos fornecidos em CONTEXT.
Se a resposta não estiver clara nas fontes, diga explicitamente o que falta e não invente.
Explique os passos e use LaTeX quando útil.
Se houver cálculo, descreva o método e proponha um bloco ```python``` com SciPy/Sympy.
Ao final, liste as fontes consultadas com [arquivo p.X]."""

def build_prompt(question: str, retrieved_docs: List[Document], max_ctx_chars=6000) -> str:
    context_chunks = []
    total = 0
    for d in retrieved_docs:
        piece = f"[{d.metadata.get('source','?')} p.{d.metadata.get('page','?')}]\n{d.page_content.strip()}\n"
        if total + len(piece) > max_ctx_chars:
            break
        context_chunks.append(piece)
        total += len(piece)
    context = "\n---\n".join(context_chunks) if context_chunks else "(sem contexto)"
    prompt = f"""{SYSTEM_GUARDRAILS}

CONTEXT:
{context}

PERGUNTA:
{question}

Responda em português. Se necessário, inclua um bloco ```python``` seguro para os cálculos.
"""
    return prompt

# ===================== Google Drive Sync =====================
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

def _build_drive_service():
    """Cria o client do Drive com credenciais da Service Account.

    Prioriza:
    - st.secrets["GOOGLE_SERVICE_ACCOUNT_JSON"] (Streamlit Cloud)
    - env GOOGLE_SERVICE_ACCOUNT_FILE -> caminho para o JSON (local)
    """
    # Tentativa via Streamlit secrets (se disponível)
    try:
        import streamlit as st
        sa_json = st.secrets.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    except Exception:
        sa_json = None

    creds = None
    if sa_json:
        import json
        info = json.loads(sa_json)
        creds = service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
    else:
        sa_path = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE")
        if not sa_path or not os.path.exists(sa_path):
            raise RuntimeError("Credenciais não encontradas. Configure GOOGLE_SERVICE_ACCOUNT_JSON (secrets) ou GOOGLE_SERVICE_ACCOUNT_FILE (local).")
        creds = service_account.Credentials.from_service_account_file(sa_path, scopes=SCOPES)

    return build("drive", "v3", credentials=creds, cache_discovery=False)

def drive_sync_pdfs(folder_id: str, dest_dir: str) -> Tuple[int, int, bool]:
    """Baixa PDFs de uma pasta do Drive para dest_dir.
    Retorna: (baixados, pulados, houve_mudanca)
    - Usa md5Checksum para evitar re-download.
    """
    os.makedirs(dest_dir, exist_ok=True)
    service = _build_drive_service()

    q = f"'{folder_id}' in parents and mimeType='application/pdf' and trashed=false"
    page_token = None
    downloaded = 0
    skipped = 0
    changed = False

    while True:
        resp = service.files().list(
            q=q,
            spaces="drive",
            fields="nextPageToken, files(id, name, md5Checksum, modifiedTime)",
            pageToken=page_token
        ).execute()
        for f in resp.get("files", []):
            name = f["name"]
            fid = f["id"]
            md5 = f.get("md5Checksum")
            local_path = os.path.join(dest_dir, name)

            # Se já existe e bate o md5, pula
            if os.path.exists(local_path) and md5 and _local_md5(local_path) == md5:
                skipped += 1
                continue

            # baixa
            req = service.files().get_media(fileId=fid)
            with io.FileIO(local_path, "wb") as out_f:
                downloader = MediaIoBaseDownload(out_f, req)
                done = False
                while not done:
                    status, done = downloader.next_chunk()

            downloaded += 1
            changed = True

        page_token = resp.get("nextPageToken", None)
        if page_token is None:
            break

    return downloaded, skipped, changed

def _local_md5(path: str) -> str | None:
    import hashlib
    try:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None
