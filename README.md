# RAG Chatbot — PDF Q&A with Gemini + FAISS

A production-ready Retrieval-Augmented Generation (RAG) chatbot built as a B.Tech CSE project.
Upload any PDF and ask questions — answers are grounded in your document using Google Gemini.

---

## Tech Stack

| Component | Library | Version |
|---|---|---|
| UI | Streamlit | 1.45.1 |
| RAG Orchestration | LangChain | 0.3.25 |
| LLM | Google Gemini (`gemini-1.5-flash`) | via langchain-google-genai |
| Vector Database | FAISS (CPU) | 1.11.0 |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` | runs fully locally |
| PDF Parsing | PyPDF | 5.4.0 |

---

## Project Structure

```
rag-chatbot/
│
├── app.py                   # Main Streamlit UI and session management
├── requirements.txt         # All Python dependencies (pinned versions)
├── .env                     # Your API key goes here (create this yourself)
├── .env.example             # Template showing what .env should look like
├── README.md
│
├── data/                    # Drop test PDFs here
├── vectorstore/             # FAISS index auto-saved here after processing
│
└── utils/
    ├── __init__.py
    ├── pdf_loader.py        # Loads PDFs from disk or Streamlit upload
    ├── text_splitter.py     # Splits documents into overlapping chunks
    ├── embeddings.py        # Loads HuggingFace sentence-transformer model
    ├── vector_store.py      # FAISS: create, save, load
    └── rag_chain.py         # Gemini LLM + retrieval + answer generation
```

---

## Installation & Setup

### 1. Download or clone the project

```bash
git clone <your-repo-url>
cd rag-chatbot
```

### 2. Create a virtual environment

```bash
# Linux / macOS
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

You should see `(venv)` appear in your terminal prompt.

### 3. Install all dependencies

```bash
pip install -r requirements.txt
```

> The first run will also download the `all-MiniLM-L6-v2` embedding model (~90 MB) from HuggingFace automatically and cache it locally. This only happens once.

### 4. Get a Gemini API Key

1. Open **[aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)** in your browser
2. Sign in with your Google account
3. Click **"+ Create API key"**
4. Select a Google Cloud project from the dropdown
5. Click **"Create API key in existing project"**
6. **Copy the key** — it may start with `AIzaSy...` or `AQ.` depending on your account

> **Test your key in terminal before using the app:**
> ```bash
> curl "https://generativelanguage.googleapis.com/v1beta/models?key=YOUR_KEY_HERE"
> ```
> If you see a list of model names, your key is valid and ready to use.

### 5. Create your `.env` file

```bash
# Linux / macOS
cp .env.example .env

# Windows
copy .env.example .env
```

Open `.env` and replace the placeholder with your actual key:

```
GOOGLE_API_KEY=YOUR_KEY_HERE
```

> Common mistakes:
> - File must be named `.env` not `.env.example`
> - No quotes around the key: `KEY=value` not `KEY="value"` ❌
> - No spaces around `=`: `KEY=value` not `KEY = value` ❌
> - The `.env` file must be in the same folder as `app.py`

Verify it is loaded correctly:
```bash
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print(os.getenv('GOOGLE_API_KEY'))"
```
This should print your key. If it prints `None`, the `.env` file is missing or in the wrong location.

### 6. Run the app

```bash
streamlit run app.py
```

The app opens automatically at **http://localhost:8501**

---

## How to Use

1. **Upload a PDF** — Click "Browse files" in the sidebar and select any PDF
2. **Click "Process PDF"** — The app runs the full pipeline: load → split → embed → index
3. **Ask questions** — Type in the chat box at the bottom of the page
4. **View sources** — Expand "Source references" under any answer to see which pages were used
5. **Load saved index** — On future runs, click "Load saved index" to skip reprocessing

---

## How the RAG Pipeline Works

```
PDF Upload
    │
    ▼
PyPDF extracts text  →  List of page Documents
    │
    ▼
RecursiveCharacterTextSplitter  →  ~chunks of 1000 chars (200 overlap)
    │
    ▼
all-MiniLM-L6-v2 (local)  →  384-dimensional vectors
    │
    ▼
FAISS index  →  saved to vectorstore/faiss_index/
    │
    ▼
User asks a question
    │
    ▼
FAISS retrieves top-4 most similar chunks
    │
    ▼
Gemini 1.5 Flash generates answer grounded in those chunks
    │
    ▼
Answer + source page references shown in chat UI
```

The embedding model runs **100% locally** — only answer generation uses the Gemini API.

---

## Configuration Reference

| Setting | File | Default |
|---|---|---|
| Chunk size | `utils/text_splitter.py` | 1000 characters |
| Chunk overlap | `utils/text_splitter.py` | 200 characters |
| Top-k retrieval | `utils/rag_chain.py` | 4 chunks |
| Gemini model | `utils/rag_chain.py` | `gemini-1.5-flash` |
| Embedding model | `utils/embeddings.py` | `all-MiniLM-L6-v2` |
| FAISS save path | `utils/vector_store.py` | `vectorstore/faiss_index` |

---

## Troubleshooting

**`GOOGLE_API_KEY is not set` or `None`**
The `.env` file is missing or in the wrong folder. It must be in the same directory as `app.py`. Run `ls -la` to check it exists.

**`API key not valid` / `API_KEY_INVALID`**
Your key is wrong or expired. Test it with the `curl` command shown in Step 4 above. If it fails there too, delete the key on AI Studio and create a fresh one.