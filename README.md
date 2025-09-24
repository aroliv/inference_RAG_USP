# 📚 Inference RAG USP

Aplicação de **Retrieval-Augmented Generation (RAG)** para apoiar os estudos de **Inferência Estatística**.  
Desenvolvida em **Streamlit** usando **Gemini (Google)** e sincronização automática dos PDFs da disciplina a partir de uma **pasta no Google Drive**.

---

## ✨ Funcionalidades

- 🔄 Sincronização automática dos slides em PDF de uma pasta única do Google Drive.  
- 🧠 Indexação com **FAISS + embeddings Google**.  
- ❓ Perguntas teóricas ou com cálculos estatísticos.  
- 📑 Respostas ancoradas nos slides, com **citações de página e arquivo**.  
- 🧮 Execução opcional de cálculos sugeridos (via `numpy`, `scipy`, `sympy`).  

---

## 🚀 Deploy no Streamlit Cloud

1. Faça fork ou clone deste repositório.  
2. No [Streamlit Cloud](https://share.streamlit.io/) crie um novo app:  
   - **Repository:** este repo  
   - **Branch:** `main`  
   - **Main file path:** `streamlit_app.py`  
3. Em **Settings → Secrets**, adicione o seguinte conteúdo (ajuste com suas credenciais):

```toml
GOOGLE_API_KEY = "SUA_CHAVE_DO_GEMINI"

MODEL_NAME = "gemini-1.5-pro"
EMBED_MODEL = "text-embedding-004"

GOOGLE_DRIVE_FOLDER_ID = "1t8TgCrN4A1zP6dUdgU2gMc-EOMYlcSbM"

# Cole o CONTEÚDO COMPLETO do JSON da Service Account (Drive API ativada)
GOOGLE_SERVICE_ACCOUNT_JSON = """
{
  "type": "service_account",
  "project_id": "...",
  "private_key_id": "...",
  "private_key": "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n",
  "client_email": "xxx@yyy.iam.gserviceaccount.com",
  "client_id": "...",
  ...
}
"""
```
---

# ⚠️ **Importante**: compartilhe a pasta do Google Drive com o e-mail da Service Account (client_email do JSON) com permissão Leitor.

---

## 📂 **Estrutura do projeto**
```
inference_RAG_USP/
├── streamlit_app.py       # App principal (interface + consultas)
├── rag_utils.py           # Funções auxiliares (RAG + sync Drive)
├── requirements.txt       # Dependências do projeto
├── runtime.txt            # Define versão do Python (3.11)
├── README.md              # Este arquivo
└── .gitignore             # Ignora env, índice, PDFs locais, etc.
```

Pastas criadas automaticamente em runtime:
```
data/
 ├── slides/   # PDFs baixados do Google Drive
 └── index/    # Índice FAISS gerado
```
---
## 🔧 Tecnologias utilizadas

- Streamlit → interface web interativa
- LangChain → integração embeddings + FAISS
- Google Generative AI → Gemini API
- FAISS → busca vetorial
- SciPy, NumPy, Sympy → cálculos estatísticos
- Google Drive API → sincronização de PDFs

---
## 📖 Como usar

1. Abra o app publicado no Streamlit Cloud (link do deploy).
2. O app sincroniza automaticamente os PDFs da pasta configurada no Drive.
3. Digite sua pergunta:
   - Exemplo teórico:
     ```"Qual a diferença entre teste unilateral e bilateral?"```
   - Exemplo com cálculo:
     ```"Construa o IC 95% para a média amostral 12.4, n=25, s=3.2."```
4. A resposta traz explicações detalhadas com citações aos slides.
5. Se houver bloco python sugerido, clique em Executar para ver o cálculo.

---

# ⚠️ Aviso
Este app foi feito para apoio ao estudo.
Use de forma ética e conforme as regras da disciplina. 🚫📄
