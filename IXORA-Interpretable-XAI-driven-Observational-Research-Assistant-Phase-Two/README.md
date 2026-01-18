# IXORA - Interpretable XAI-driven Observational Research Assistant

## Overview
IXORA is a biomedical research assistant that leverages Explainable AI (XAI) to provide interpretable insights for observational medical research. It combines state-of-the-art language models with causal inference techniques to help researchers analyze biomedical data and literature more effectively.

## Key Features

### Core Functionality
- **Interactive Chat Interface**: Natural language interface for research queries
- **Causal Inference**: On-demand causal analysis of research questions
- **Literature Review**: Integration with arXiv for evidence-based research
- **Parameter Extraction**: Automated extraction of biomedical parameters
- **Model Interpretability**: SHAP-based explanations for model predictions

### Technical Components
- **Backend**: FastAPI-based RESTful API service
- **Frontend**: Streamlit-based web interface
- **Machine Learning**: Integration with Mistral and other language models
- **Data Processing**: Tools for handling biomedical data and research papers
- **RLHF Integration**: Reinforcement Learning from Human Feedback for model improvement

## Project Structure

```
.
├── core/                     # Backend core functionality
│   ├── analytics.py         # Causal and SHAP analysis
│   ├── arxiv.py             # arXiv paper retrieval
│   ├── biomed_parameter_extractor.py  # Parameter extraction
│   ├── celery_app.py        # Async task queue
│   ├── config.py            # Configuration settings
│   ├── langgraph.py         # Multi-agent system
│   ├── main.py              # Main FastAPI application
│   ├── medicalscience/      # Medical science modules
│   ├── mistral.py           # Mistral model integration
│   ├── model_loader.py      # Model loading utilities
│   ├── prompts.py           # System and user prompts
│   ├── rlhf/                # RLHF implementation
│   └── utils.py             # Utility functions
├── frontend/                # Streamlit web interface
│   ├── app.py               # Main application
│   └── requirements-streamlit.txt  # Frontend dependencies
├── tests/                   # Test files
└── api_response.json        # Example API responses
```

## Installation

### Prerequisites
- Python 3.8+
- pip
- Redis (for task queue)
- CUDA-compatible GPU (recommended)

### Setup
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd IXORA-Interpretable-XAI-driven-Observational-Research-Assistant-Phase-Two
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. Install backend dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install frontend dependencies:
   ```bash
   cd frontend
   pip install -r requirements-streamlit.txt
   ```

## Configuration

1. Copy `.env.example` to `.env` and update the following variables:
   ```
   # API Configuration
   BASE_API_URL=http://localhost:8000
   
   # Model Configuration
   MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.1
   
   # Redis Configuration (for Celery)
   REDIS_URL=redis://localhost:6379/0
   ```

## Running the Application

### Start the Backend
```bash
cd core
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Start the Frontend
```bash
cd frontend
streamlit run app.py
```

### Start Celery Worker (for async tasks)
```bash
celery -A core.celery_app worker --loglevel=info
```

## Usage

1. Access the web interface at `http://localhost:8501`
2. Enter your research question in the chat interface
3. Use the sidebar to access additional features:
   - Causal analysis
   - Literature review
   - Model explanations
   - Feedback system

## API Endpoints

- `POST /chat`: Main chat endpoint
- `POST /causal`: Causal analysis endpoint
- `POST /arxiv`: Literature search endpoint
- `POST /feedback`: Submit feedback for RLHF
- `GET /health`: Health check

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use IXORA in your research, please cite:

```
@misc{ixora2024,
  title={IXORA: Interpretable XAI-driven Observational Research Assistant},
  author={IXORA Team},
  year={2024},
  publisher={GitHub},
  howpublished={\url{https://github.com/yourusername/ixora}}
}
```

## Support

For support, please open an issue in the GitHub repository or contact the maintainers.
