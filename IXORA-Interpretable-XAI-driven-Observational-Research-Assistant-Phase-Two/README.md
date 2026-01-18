# 🔬 IXORA - Interpretable XAI-driven Observational Research Assistant

> **An intelligent biomedical research assistant that combines Explainable AI with causal inference to provide interpretable insights for observational medical research.**

![Python Version](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Phase](https://img.shields.io/badge/Phase-Two-orange.svg)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Architecture](#architecture)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)
- [Support](#support)

---

## 🎯 Overview

IXORA is a cutting-edge biomedical research assistant designed specifically for observational research. It leverages **Explainable AI (XAI)** methodologies to transform complex medical data and literature into actionable, interpretable insights. By combining state-of-the-art language models (Mistral, Qwen) with causal inference techniques, IXORA empowers researchers to:

- Analyze biomedical datasets with built-in interpretability
- Extract insights from medical literature automatically
- Generate evidence-based research recommendations
- Understand model predictions through SHAP-based explanations
- Improve model performance through human feedback (RLHF)

---

## ✨ Key Features

### 🧠 Core Functionality
- **Interactive Chat Interface**: Natural language interface for intuitive research queries
- **Causal Inference Engine**: Advanced causal analysis for understanding research relationships
- **Literature Integration**: Automated arXiv paper retrieval and analysis
- **Smart Parameter Extraction**: Biomedical-specific parameter extraction from documents
- **Model Interpretability**: SHAP-based visual explanations for all predictions
- **Feedback Loop**: Reinforcement Learning from Human Feedback (RLHF) for continuous improvement

### 🏗️ Technical Stack
- **Backend**: FastAPI with asynchronous support
- **Frontend**: Streamlit web interface with real-time updates
- **LLMs**: Integration with Mistral, Qwen, and other state-of-the-art models
- **Task Queue**: Celery with Redis for distributed processing
- **Analytics**: SHAP, scikit-learn for interpretability and analysis
- **NLP**: Transformers, NLTK, and sentence-transformers

### 🔄 Specialized Modules
- **Medical Science Pipeline**: Domain-specific processing for medical literature
- **Computer Science Pipeline**: Adapters for computer science research papers
- **RLHF System**: Reward modeling and trainer for human feedback integration

---

## 📁 Project Structure

```
IXORA/
├── 📂 core/                              # Backend core functionality
│   ├── main.py                           # FastAPI application entry point
│   ├── config.py                         # Configuration management
│   ├── celery_app.py                     # Async task queue setup
│   │
│   ├── 🤖 Model Integration
│   ├── mistral.py                        # Mistral model wrapper
│   ├── model_loader.py                   # Model loading utilities
│   ├── prompts.py                        # System & user prompts library
│   │
│   ├── 📊 Analytics & Interpretability
│   ├── analytics.py                      # SHAP analysis & causal inference
│   ├── biomed_parameter_extractor.py     # Medical parameter extraction
│   ├── cs_parameter_extractor.py         # CS-specific extraction
│   │
│   ├── 🔗 LangGraph & Multi-Agent
│   ├── langgraph.py                      # Multi-agent orchestration
│   ├── arxiv.py                          # arXiv paper retrieval
│   ├── utils.py                          # Utility functions
│   │
│   ├── 📚 Domain-Specific Modules
│   ├── medicalscience/
│   │   ├── loaders.py                    # Data loaders for medical data
│   │   ├── pipeline.py                   # Medical science pipeline
│   │   └── state.py                      # State management
│   │
│   ├── computerscience/
│   │   ├── loaders.py                    # Data loaders for CS papers
│   │   ├── pipeline.py                   # CS research pipeline
│   │   └── state.py                      # State management
│   │
│   └── 🎓 RLHF System
│       └── rlhf/
│           ├── __init__.py
│           ├── reward_model.py           # Reward model training
│           ├── trainer.py                # RLHF trainer
│           └── feedback_logger.py        # User feedback logging
│
├── 📂 frontend/                          # Streamlit web interface
│   ├── app.py                            # Main Streamlit application
│   └── requirements-streamlit.txt        # Frontend dependencies
│
├── 🧪 Test Files
│   ├── test_backend.py                   # Backend API tests
│   ├── test_biomistral.py                # Biomedical model tests
│   ├── test_computerscience_mistral.py   # CS model tests
│   ├── test_simple_mistral.py            # Basic model tests
│   └── diagnostic_test.py                # Diagnostic tests
│
├── 📄 Configuration Files
│   ├── requirements.txt                  # Backend dependencies
│   ├── .gitignore                        # Git ignore rules
│   ├── api_response.json                 # Example API responses
│   ├── setup_nltk.py                     # NLTK data setup
│   └── README.md                         # This file
│
└── Debug Files
    ├── debug_api.py                      # API debugging script
    └── debug_mistral_raw.py              # Model debugging script
```

---

## 🚀 Installation

### Prerequisites

Before installation, ensure you have the following:

- **Python 3.8 or higher**
- **pip** (Python package manager)
- **Git**
- **Redis** (for Celery task queue) - [Installation Guide](https://redis.io/download)
- **CUDA-compatible GPU** (recommended for faster inference)
- **4GB+ RAM** (minimum), **8GB+** (recommended)

### Quick Start

#### Step 1: Clone the Repository
```bash
git clone https://github.com/SrishtiTurki/IXORA-Interpretable-XAI-driven-Observational-Research-Assistant.git
cd IXORA-Interpretable-XAI-driven-Observational-Research-Assistant-Phase-Two
```

#### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

#### Step 3: Install Dependencies
```bash
# Backend dependencies
pip install -r requirements.txt

# Frontend dependencies
pip install -r frontend/requirements-streamlit.txt

# NLTK data setup
python setup_nltk.py
```

#### Step 4: Verify Installation
```bash
# Test backend import
python -c "import fastapi; print('✓ FastAPI installed')"

# Test Torch
python -c "import torch; print('✓ PyTorch installed')"

# Test Transformers
python -c "import transformers; print('✓ Transformers installed')"
```

---

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root with the following variables:

```bash
# API Configuration
BASE_API_URL=http://localhost:8000
API_HOST=0.0.0.0
API_PORT=8000

# Model Configuration
MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.1
MODEL_DEVICE=cuda  # Use 'cuda' for GPU, 'cpu' for CPU
MAX_TOKENS=1024
TEMPERATURE=0.7

# arXiv Configuration
ARXIV_SEARCH_LIMIT=10
ARXIV_TIMEOUT=30

# Redis Configuration (for Celery)
REDIS_URL=redis://localhost:6379/0
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0

# Logging Configuration
LOG_LEVEL=INFO
DEBUG_MODE=False

# RLHF Configuration
ENABLE_RLHF=True
FEEDBACK_STORAGE_PATH=./feedback_logs
```

### Using Default Config

If you don't create a `.env` file, the application uses sensible defaults from `core/config.py`.

---

## 🎮 Usage

### Starting the Application

#### Option 1: Full Stack (Backend + Frontend)

**Terminal 1 - Start Backend:**
```bash
cd core
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Output:
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

**Terminal 2 - Start Redis (if needed):**
```bash
redis-server
```

**Terminal 3 - Start Celery Worker:**
```bash
celery -A core.celery_app worker --loglevel=info
```

**Terminal 4 - Start Frontend:**
```bash
cd frontend
streamlit run app.py
```

Access the web interface at: **http://localhost:8501**

#### Option 2: Backend Only (API Testing)

```bash
cd core
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Test with cURL:
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the effect of exercise on heart disease?"}'
```

### Using the Web Interface

1. **Open Streamlit App**: Navigate to `http://localhost:8501`
2. **Ask a Research Question**: Type your biomedical research question
3. **Select Features**:
   - 📊 **Causal Analysis**: Understand causal relationships
   - 📚 **Literature Review**: Fetch relevant papers from arXiv
   - 🎯 **Parameter Extraction**: Extract biomedical parameters
   - 💡 **Model Explanations**: View SHAP-based explanations
4. **Provide Feedback**: Rate responses to improve the model

---

## 🔌 API Documentation

### Chat Endpoint

**Request:**
```bash
POST /chat
Content-Type: application/json

{
  "question": "What is the relationship between smoking and lung cancer?",
  "research_type": "medical"  # or "computer-science"
}
```

**Response:**
```json
{
  "answer": "Based on observational research...",
  "confidence": 0.92,
  "sources": ["arXiv:2024.xxxxx", "arXiv:2024.yyyyy"],
  "explanation": "..."
}
```

### Causal Analysis Endpoint

**Request:**
```bash
POST /causal
Content-Type: application/json

{
  "treatment": "Exercise",
  "outcome": "Heart Disease",
  "confounders": ["Age", "BMI", "Smoking"]
}
```

**Response:**
```json
{
  "causal_effect": 0.35,
  "confidence_interval": [0.25, 0.45],
  "method": "doubly_robust"
}
```

### Literature Search Endpoint

**Request:**
```bash
POST /arxiv
Content-Type: application/json

{
  "query": "causal inference medical research",
  "max_results": 10
}
```

**Response:**
```json
{
  "papers": [
    {
      "title": "...",
      "authors": ["..."],
      "url": "https://arxiv.org/...",
      "summary": "..."
    }
  ]
}
```

### Feedback Endpoint

**Request:**
```bash
POST /feedback
Content-Type: application/json

{
  "query": "...",
  "response": "...",
  "rating": 5,
  "comments": "..."
}
```

### Health Check

```bash
GET /health
```

Response: `{"status": "healthy"}`

---

## 🏛️ Architecture

### System Architecture

```
┌─────────────────────────────────────────────┐
│        Streamlit Web Interface              │
│     (http://localhost:8501)                 │
└──────────────────┬──────────────────────────┘
                   │
                   ↓
┌─────────────────────────────────────────────┐
│      FastAPI Backend                        │
│   (http://localhost:8000)                   │
├─────────────────────────────────────────────┤
│ ┌─────────────────────────────────────────┐ │
│ │    Core Processing Layer                │ │
│ ├─────────────────────────────────────────┤ │
│ │ • Mistral/Qwen LLM Integration         │ │
│ │ • Parameter Extraction                  │ │
│ │ • SHAP Analysis                         │ │
│ │ • Causal Inference                      │ │
│ │ • arXiv Integration                     │ │
│ └─────────────────────────────────────────┘ │
└──────────────────┬──────────────────────────┘
                   │
        ┌──────────┼──────────┐
        ↓          ↓          ↓
   ┌────────┐ ┌────────┐ ┌────────┐
   │ Redis  │ │ Models │ │ NLTK   │
   │ Celery │ │ Cache  │ │ Data   │
   └────────┘ └────────┘ └────────┘
```

### Data Flow

1. **User Query** → Streamlit UI
2. **API Request** → FastAPI Backend
3. **Processing** → LLM + Analytics
4. **Analysis** → SHAP Explanations + Causal Inference
5. **Literature** → arXiv Search
6. **Response** → Rendered UI
7. **Feedback** → RLHF Training Loop

---

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run specific test file
pytest test_biomistral.py -v

# Run with coverage
pytest --cov=core

# Run diagnostic tests
python diagnostic_test.py
```

---

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. **Fork the Repository**
   ```bash
   git clone https://github.com/SrishtiTurki/IXORA-Interpretable-XAI-driven-Observational-Research-Assistant.git
   ```

2. **Create a Feature Branch**
   ```bash
   git checkout -b feature/YourFeatureName
   ```

3. **Make Your Changes**
   - Follow PEP 8 style guidelines
   - Add tests for new features
   - Update documentation

4. **Commit Your Changes**
   ```bash
   git commit -m "Add: description of your feature"
   ```

5. **Push to Branch**
   ```bash
   git push origin feature/YourFeatureName
   ```

6. **Open a Pull Request**
   - Describe your changes
   - Link related issues
   - Request review from maintainers

### Code Style

We use:
- **Black** for code formatting
- **isort** for import sorting
- **pylint** for linting
- **mypy** for type checking

```bash
# Format code
black core/ frontend/

# Sort imports
isort core/ frontend/

# Lint
pylint core/

# Type check
mypy core/
```

---

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

You are free to:
- ✅ Use this project for any purpose
- ✅ Modify and distribute
- ✅ Use commercially

With the condition of including the license notice.

---

## 📚 Citation

If you use IXORA in your research, please cite:

**BibTeX:**
```bibtex
@misc{ixora2024,
  title={IXORA: Interpretable XAI-driven Observational Research Assistant - Phase Two},
  author={Srishti Turki},
  year={2024},
  publisher={GitHub},
  howpublished={\url{https://github.com/SrishtiTurki/IXORA-Interpretable-XAI-driven-Observational-Research-Assistant}}
}
```

**APA:**
```
Turki, S. (2024). IXORA: Interpretable XAI-driven Observational Research Assistant - Phase Two. GitHub. Retrieved from https://github.com/SrishtiTurki/IXORA-Interpretable-XAI-driven-Observational-Research-Assistant
```

---

## 🆘 Support & Documentation

### Troubleshooting

**Issue: CUDA Out of Memory**
```bash
# Use CPU instead
export TORCH_DEVICE=cpu
python -c "from core.model_loader import load_model; load_model(device='cpu')"
```

**Issue: Redis Connection Error**
```bash
# Ensure Redis is running
redis-server
# or use Docker
docker run -d -p 6379:6379 redis
```

**Issue: NLTK Data Missing**
```bash
python setup_nltk.py
```

### Documentation

- 📖 **Full Documentation**: [docs/](docs/) (coming soon)
- 🐛 **Issues & Bugs**: [GitHub Issues](https://github.com/SrishtiTurki/IXORA-Interpretable-XAI-driven-Observational-Research-Assistant/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/SrishtiTurki/IXORA-Interpretable-XAI-driven-Observational-Research-Assistant/discussions)

### Getting Help

1. **Check Existing Issues**: Search [GitHub Issues](https://github.com/SrishtiTurki/IXORA-Interpretable-XAI-driven-Observational-Research-Assistant/issues)
2. **Create New Issue**: Provide:
   - System info (OS, Python version)
   - Error messages & logs
   - Steps to reproduce
3. **Contact Maintainers**: Open a discussion for general questions

---

## 🙏 Acknowledgments

- **Mistral AI** for providing the Mistral models
- **OpenAI** & **Meta** for model inspiration
- **arXiv** for paper access API
- **SHAP** community for interpretability tools
- All contributors and users for feedback

---

## 📞 Contact

- **Author**: Srishti Turki
- **GitHub**: [@SrishtiTurki](https://github.com/SrishtiTurki)
- **Repository**: [IXORA](https://github.com/SrishtiTurki/IXORA-Interpretable-XAI-driven-Observational-Research-Assistant)

---

**Last Updated**: January 18, 2026 | **Phase**: Two | **Status**: Active Development
