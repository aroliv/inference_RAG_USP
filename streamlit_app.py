import os
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
    render_citations, might_need_calc,
    drive_sync_pdfs
)

INDEX_DIR = "data/index"
SLIDES_DIR = "data/slides"
os.makedirs(SLIDES_DIR, exist_ok=True)

st.set_page_config(page_title="RAG - Inferência Estatística", layout="wide")
st.title("📚 RAG para Inferência Estatística (Drive Sync)")

# --------- Credenciais / Config ---------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID") or st.secrets.get("GOOGLE_DRIVE_FOLDER_ID")
if not FOLDER_ID:
    st.warning("Configure GOOGLE_DRIVE_FOLDER_ID (env ou secrets).")
    st.stop()

model_name = os.getenv("MODEL_NAME", "gemini-1.5-pro")
llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.2)

# --------- Sync do Drive ao iniciar ---------
with st.spinner("Sincronizando PDFs do Google Drive..."):
    try:
        downloaded, skipped, changed = drive_sync_pdfs(FOLDER_ID, SLIDES_DIR)
        st.info(f"Drive: baixados {downloaded}, pulados {skipped}.")
    except Exception as e:
        st.error(f"Falha ao sincronizar com o Drive: {e}")
        st.stop()

# --------- (Re)indexação condicional ---------
def index_exists() -> bool:
    return os.path.exists(os.path.join(INDEX_DIR, "index.faiss")) or any(
        f.endswith(".faiss") for f in os.listdir(INDEX_DIR)) if os.path.isdir(INDEX_DIR) else False

need_reindex = changed or (not index_exists())
if need_reindex:
    with st.spinner("Construindo índice FAISS a partir dos PDFs..."):
        pdf_paths = [os.path.join(SLIDES_DIR, f) for f in os.listdir(SLIDES_DIR) if f.endswith(".pdf")]
        if not pdf_paths:
            st.warning("Nenhum PDF encontrado após sync. Verifique a pasta do Drive.")
            st.stop()
        docs = pdfs_to_documents(pdf_paths)
        chunks = chunk_documents(docs, chunk_size=1200, chunk_overlap=160)
        build_vectorstore(chunks, INDEX_DIR)
        st.success("Índice criado/atualizado!")

# Carrega o índice
try:
    vs = load_vectorstore(INDEX_DIR)
except Exception as e:
    st.error(f"Erro ao carregar índice: {e}")
    st.stop()

# --------- UI principal ---------
st.sidebar.header("Opções")
k = st.sidebar.slider("k (documentos)", min_value=2, max_value=8, value=4, step=1)
score_threshold = st.sidebar.slider("limiar de similaridade (0=mais amplo)", 0.0, 1.0, 0.0, 0.05)
do_calc = st.sidebar.checkbox("Permitir execução de cálculo Python sugerido", value=True)

if st.sidebar.button("Sincronizar Drive + Reindexar agora"):
    st.experimental_rerun()

question = st.text_area("Digite sua pergunta (teórica ou com números):", height=120,
                        placeholder="Ex.: Como construir um IC 95% para a média com variância desconhecida?")

if st.button("Consultar"):
    if not question.strip():
        st.warning("Digite uma pergunta.")
        st.stop()

    retrieved, scores = vs.similarity_search_with_score(question, k=k)
    if score_threshold > 0:
        pairs = [(d, s) for d, s in zip(retrieved, scores) if s <= score_threshold]
        if pairs:
            retrieved, _ = zip(*pairs)
            retrieved = list(retrieved)

    prompt = build_prompt(question, retrieved_docs=retrieved)
    messages = [
        SystemMessage(content="Você é um tutor de Inferência Estatística que dá respostas ancoradas em fontes."),
        HumanMessage(content=prompt),
    ]

    with st.spinner("Pensando com base nos slides..."):
        resp = llm(messages)
    answer = resp.content

    st.markdown("### Resposta")
    st.markdown(answer)
    st.markdown(render_citations(retrieved))

    code = extract_python_block(answer)
    if do_calc and code:
        st.markdown("#### Executar cálculo sugerido")
        if st.button("Rodar bloco ```python``` com SciPy/Sympy"):
            ok, out = run_calc_block(code)
            if ok:
                st.success("Cálculo executado.")
                st.code(out)
            else:
                st.error(out)
    else:
        if might_need_calc(question) and not code:
            st.info("Parece haver cálculo envolvido. Peça: 'inclua um bloco ```python``` com os cálculos'.")
