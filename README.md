# 🌸 IXORA — Interpretable XAI-driven Observational Research Assistant

> An AI-powered research platform that combines **Explainable AI (XAI)**, **causal inference**, and **LLM-based reasoning** to make scientific literature more interpretable and trustworthy.

---

## 🧠 What is IXORA?

IXORA is a full-stack research assistant built for researchers, clinicians, and data scientists who need more than just search — they need **understanding**. By integrating state-of-the-art XAI techniques (SHAP, LIME), causal modeling (DoWhy, EconML), and large language models (Mistral AI, LangChain), IXORA enables users to ask complex research questions and receive interpretable, evidence-backed answers grounded in academic literature.

---

## ✨ Features

- **Explainable AI Integration** — SHAP and LIME explanations for model predictions, giving users insight into *why* an answer was generated
- **Causal Inference Engine** — DoWhy and EconML power causal analysis beyond correlation, enabling observational research workflows
- **LLM-Powered Q&A** — Mistral AI + LangChain + LangGraph orchestrate multi-step reasoning over retrieved scientific papers
- **ArXiv Paper Retrieval** — Automatically fetches and parses relevant research papers using the `arxiv` client and PyMuPDF
- **Semantic Search** — Sentence Transformers enable dense vector search over ingested literature
- **Async Task Processing** — Celery + Redis handle long-running analytics jobs without blocking the API
- **Authentication** — Firebase + JWT-based auth with bcrypt password hashing
- **REST API** — FastAPI backend with separate auth service (ports 8000 & 8001)
- **Web Frontend** — Vite-powered frontend with Firebase integration

---

## 🗂️ Project Structure

```
IXORA/
├── core/                   # FastAPI backend — main app, routes, Celery tasks
│   ├── main.py             # Primary API entry point (port 8000)
│   ├── celery_app.py       # Celery worker configuration
│   └── auth/
│       └── auth_api.py     # Auth service entry point (port 8001)
├── frontend/               # Legacy frontend
├── ixora-web/              # Vite + Firebase web app
├── Redis/                  # Redis server binary (Windows)
├── Test_backend_old/       # Archived backend for reference
├── requirements.txt        # Python dependencies
└── package.json            # Node scripts to orchestrate all services
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **LLM / Agents** | Mistral AI, LangChain, LangGraph, llama-cpp-python |
| **XAI** | SHAP, LIME |
| **Causal Inference** | DoWhy, EconML |
| **ML / NLP** | scikit-learn, Transformers (HuggingFace), Sentence Transformers |
| **Paper Retrieval** | arxiv, PyMuPDF |
| **Backend** | FastAPI, Uvicorn, Celery |
| **Cache / Queue** | Redis |
| **Database** | MongoDB |
| **Auth** | Firebase Admin, JWT, bcrypt, Passlib |
| **Frontend** | Vite, Firebase |
| **Optimization** | scikit-optimize |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Node.js 18+
- Redis (included in `/Redis` for Windows)
- MongoDB instance (local or Atlas)
- Mistral AI API key
- Firebase project credentials

### 1. Clone the repository

```bash
git clone https://github.com/SrishtiTurki/IXORA-Interpretable-XAI-driven-Observational-Research-Assistant.git
cd IXORA-Interpretable-XAI-driven-Observational-Research-Assistant
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Node dependencies (root)

```bash
npm install
```

### 4. Configure environment variables

Create a `.env` file in the root directory:

```env
MISTRAL_API_KEY=your_mistral_api_key
MONGO_URI=your_mongodb_connection_string
FIREBASE_CREDENTIALS=path/to/firebase_credentials.json
SECRET_KEY=your_jwt_secret
```

### 5. Start the backend

From the root directory, run:

```bash
npm run dev
```

This single command concurrently spins up **Redis**, the **Celery worker**, and the **FastAPI server** — everything the backend needs.

### 6. Start the frontend

In a new terminal, navigate to the `ixora-web` directory and run:

```bash
cd ixora-web
npm install
npm run dev
```

The app will be available at `http://localhost:5173` (or whichever port Vite assigns).

---

## 🔬 How It Works

1. **User submits a research query** via the web interface
2. **ArXiv retrieval** fetches relevant papers, parsed as PDFs via PyMuPDF
3. **Sentence Transformers** embed the papers for semantic search
4. **LangGraph agent** orchestrates multi-step LLM reasoning over the retrieved context
5. **SHAP / LIME** explain model confidence scores for ranked results
6. **DoWhy / EconML** optionally run causal graphs for observational study analysis
7. **Response returned** to the user with both the answer and its XAI explanation

---

## 📋 API Overview

The backend exposes two FastAPI services:

- `http://localhost:8000` — Core research API (query, retrieval, XAI, causal inference)
- `http://localhost:8001` — Authentication API (register, login, token refresh)

Interactive API docs available at `http://localhost:8000/docs` (Swagger UI).

---


*IXORA — Because research deserves to be both intelligent and explainable.*
