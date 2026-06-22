# RAG Chatbot using LangChain + Streamlit

A chatbot that lets you upload a PDF and have a conversation with it. Built with LangChain for the RAG pipeline, FAISS for vector search, and Streamlit for the UI.

**Live demo:** https://rag-chat81.streamlit.app/

---

## What it does

- Upload any PDF and the app will process it — load, chunk, embed, and store it using FAISS
- Ask questions about the document in a chat interface
- Get answers grounded in the actual document content, with source references so you can verify

## Supported LLM Providers

You can plug in whichever one you prefer:

- Ollama
- Gemini
- OpenAI
- Claude
- Groq

## How it works

1. You upload a PDF
2. The app loads the file, splits the text into chunks, and embeds each chunk using FAISS, which gets stored in a vector database
3. When you ask a question, it's matched against the stored embeddings to pull the most relevant chunks
4. The LLM uses those chunks to generate an answer
5. The chat history and source references are shown in the UI

## Tech Stack

- **Streamlit** — web interface
- **LangChain** — RAG pipeline
- **FAISS** — vector similarity search
- **Your choice of LLM** — answer generation

---

## Running locally

```bash
python3 -m venv venv
source rag-chatbot-env/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
