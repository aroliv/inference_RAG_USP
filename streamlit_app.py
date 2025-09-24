import os
import time
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage

from rag_utils import (
    load_vectorstore, build_vectorstore,
    pdfs_to_documents, chunk_documents,
    build_prompt, extract_python_block, run_calc_block,
    render_citations, might_need_calc, drive_sync_pdfs
)

# ------------------------- Configuração de página -------------------------
st.set_page_config(page_title="RAG - Inferência Estatística (Chat)", page_icon="📚", layout="wide")
st.title("📚 RAG para Inferência Estatística — Chat")

# ------------------------- Lê credenciais/IDs -------------------------
# Preferir st.secrets em cloud; usar env como fallback local
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID") or st.secrets.get("GOOGLE_DRIVE_FOLDER_ID")
if not FOLDER_ID:
    st.error("Defina GOOGLE_DRIVE_FOLDER_ID (em Secrets ou .env).")
    st.stop()

MODEL_NAME = os.getenv("MODEL_NAME", "") or st.secrets.get("MODEL_NAME", "gemini-1.5-pro")
TEMPERATURE = float(os.getenv("MODEL_TEMPERATURE", "0.2"))

# ------------------------- Caminhos de dados -------------------------
INDEX_DIR = "data/index"
SLIDES_DIR = "data/slides"
os.makedirs(SLIDES_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

# ------------------------- Opções (sidebar) -------------------------
with st.sidebar:
    st.header("Configurações")
    k = st.slider("k (documentos recuperados)", min_value=2, max_value=8, value=4, step=1)
    score_threshold = st.slider("limiar de similaridade (0 = mais amplo)", 0.0, 1.0, 0.0, 0.05)
    do_calc = st.checkbox("Permitir execução de cálculo Python sugerido", value=True)
    st.caption("Se a resposta incluir um bloco ```python```, você poderá executá-lo com NumPy/SciPy/Sympy em sandbox.")

    st.divider()
    st.subheader("Sincronização")
    if st.button("🔁 Sincronizar Drive + Reindexar"):
        st.session_state["_force_sync"] = True
        st.experimental_rerun()

# ------------------------- Sessão / estado -------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []  # [{"role": "user"/"assistant", "content": "..."}]

if "_vectorstore_ready" not in st.session_state:
    st.session_state["_vectorstore_ready"] = False

if "_last_sync_info" not in st.session_state:
    st.session_state["_last_sync_info"] = ""

# ------------------------- Funções auxiliares -------------------------
def _index_exists() -> bool:
    if not os.path.isdir(INDEX_DIR):
        return False
    files = [f for f in os.listdir(INDEX_DIR) if f.endswith(".faiss") or f == "index.faiss"]
    return len(files) > 0

def _sync_and_maybe_reindex(force: bool = False) -> None:
    """Sincroniza PDFs do Drive e reindexa se necessário ou se force=True."""
    with st.spinner("Sincronizando PDFs do Google Drive..."):
        downloaded, skipped, changed = drive_sync_pdfs(FOLDER_ID, SLIDES_DIR)
        st.session_state["_last_sync_info"] = f"Drive: baixados {downloaded}, pulados {skipped}."
    need_reindex = force or changed or (not _index_exists())
    if need_reindex:
        with st.spinner("Construindo índice FAISS a partir dos PDFs..."):
            pdf_paths = [os.path.join(SLIDES_DIR, f) for f in os.listdir(SLIDES_DIR) if f.lower().endswith(".pdf")]
            if not pdf_paths:
                st.error("Nenhum PDF encontrado após o sync. Verifique a pasta do Drive.")
                st.stop()
            docs = pdfs_to_documents(pdf_paths)
            if not docs:
                st.error("Falha ao extrair texto dos PDFs (pypdf).")
                st.stop()
            chunks = chunk_documents(docs, chunk_size=1200, chunk_overlap=160)
            build_vectorstore(chunks, INDEX_DIR)
    # marca que temos índice pronto
    st.session_state["_vectorstore_ready"] = True

def _load_vs_or_fail() -> FAISS:
    try:
        return load_vectorstore(INDEX_DIR)
    except Exception as e:
        st.error(f"Erro ao carregar índice FAISS: {e}")
        st.stop()

def _retrieve(vs: FAISS, query: str, k_val: int, thr: float):
    retrieved, scores = vs.similarity_search_with_score(query, k=k_val)
    if thr > 0:
        pairs = [(d, s) for d, s in zip(retrieved, scores) if s <= thr]
        if pairs:
            retrieved, _ = zip(*pairs)
            retrieved = list(retrieved)
    return retrieved

def _llm():
    return ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=TEMPERATURE)

# ------------------------- Sync + índice no primeiro load -------------------------
try:
    if st.session_state.get("_force_sync", False):
        _sync_and_maybe_reindex(force=True)
        st.session_state["_force_sync"] = False
    elif not st.session_state["_vectorstore_ready"]:
        _sync_and_maybe_reindex(force=False)
except Exception as e:
    st.error(f"Falha ao sincronizar com o Drive: {e}")
    st.stop()

vs = _load_vs_or_fail()

# Info de status (discreto)
if st.session_state["_last_sync_info"]:
    st.caption(st.session_state["_last_sync_info"])

# ------------------------- Render histórico de chat -------------------------
for m in st.session_state["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# ------------------------- Entrada estilo chat -------------------------
prompt = st.chat_input("Faça sua pergunta sobre inferência estatística...")
if prompt:
    # 1) Mostrar a pergunta do usuário
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2) Recuperar contexto do índice
    try:
        retrieved_docs = _retrieve(vs, prompt, k, score_threshold)
    except Exception as e:
        with st.chat_message("assistant"):
            st.error(f"Erro na recuperação: {e}")
        st.stop()

    # 3) Montar prompt ancorado e chamar LLM
    guardrailed_prompt = build_prompt(prompt, retrieved_docs=retrieved_docs)
    messages = [
        SystemMessage(content="Você é um tutor de Inferência Estatística que dá respostas ancoradas em fontes."),
        HumanMessage(content=guardrailed_prompt),
    ]

    with st.chat_message("assistant"):
        with st.spinner("Pensando com base nos slides..."):
            try:
                resp = _llm()(messages)
                answer = resp.content
            except Exception as e:
                st.error(f"Erro no LLM: {e}")
                answer = "Não consegui consultar o modelo no momento."

        # 4) Render: resposta + citações
        st.markdown(answer)
        st.markdown(render_citations(retrieved_docs))

        # 5) Execução opcional do bloco python
        code = extract_python_block(answer)
        if do_calc and code:
            st.markdown("#### ⚙️ Executar cálculo sugerido")
            if st.button("Rodar bloco ```python``` com SciPy/Sympy", key=f"run_{len(st.session_state['messages'])}"):
                ok, out = run_calc_block(code)
                if ok:
                    st.success("Cálculo executado.")
                    st.code(out)
                else:
                    st.error(out)
        else:
            if might_need_calc(prompt) and not code:
                st.info("Parece haver cálculo envolvido. Peça: “inclua um bloco ```python``` com os cálculos”.")

    # 6) Persistir no histórico
    st.session_state["messages"].append({"role": "assistant", "content": answer})

