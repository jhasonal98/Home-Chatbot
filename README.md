# 🏠 Home Chatbot - 19th Floor 2BHK Apartment

A portfolio project demonstrating RAG (Retrieval-Augmented Generation) with a Streamlit chatbot for an interactive apartment showcase. Users can explore the apartment through structured sections and ask custom questions powered by AI.

**Live Demo:** [Deploy to Streamlit Cloud](#deployment)

---

## 🎯 Features

- **Hybrid Interface**
  - 📑 Predefined room sections (Living Room, Balcony, Washrooms, etc.)
  - 💬 RAG-powered Q&A chatbot for custom questions
  - 📚 Real-time source attribution (shows retrieved context)

- **RAG Pipeline**
  - 🤖 HuggingFace Inference API for LLM (free tier)
  - 📚 FAISS local vector database (no external DB required)
  - 🔍 Sentence-Transformers for embeddings
  - ⚡ Fast retrieval (~100ms per query)

- **Free Deployment**
  - Streamlit Community Cloud (zero cost hosting)
  - All dependencies are free/open-source
  - No database subscription required

---

## 🏗️ Project Structure

```
Home-Chatbot/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── .gitignore                  # Git ignore rules
│
├── src/
│   ├── config.py              # Configuration (models, paths, room descriptions)
│   ├── rag.py                 # RAG pipeline (embeddings, FAISS, retrieval)
│   └── llm.py                 # LLM integration (HuggingFace inference)
│
├── docs/
│   └── house_details.md       # House documentation (knowledge base)
│
├── vectorstore/               # FAISS index (generated on first run)
│   ├── index.faiss
│   └── index.pkl
│
└── .streamlit/
    ├── config.toml            # Streamlit UI configuration
    ├── secrets.template.toml  # Secrets template (copy and fill in)
    └── secrets.toml           # Actual secrets (DO NOT COMMIT)
```

---

## ⚡ Quick Start (Local)

### 1. Clone and Setup

```bash
cd Home-Chatbot

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Add HuggingFace API Token

1. Get a **free** token from [HuggingFace](https://huggingface.co/settings/tokens)
   - No credit card required
   - Free tier has rate limits (~15 requests/min)
   
2. Create `.streamlit/secrets.toml`:
   ```toml
   HF_API_TOKEN = "hf_xxxxxxxxxxxxx"
   ```

### 3. Run Locally

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

**On first run:** FAISS index will be created (~10-30 seconds). Subsequent runs are instant.

---

## 🧪 Testing the RAG Pipeline

### Test RAG Retrieval

```python
from src.rag import get_rag_pipeline

pipeline = get_rag_pipeline()

# Test retrieval
context, metadata = pipeline.retrieve("How big is the living room?", k=3)
for i, chunk in enumerate(context, 1):
    print(f"\nSource {i}:\n{chunk}")
```

### Test LLM Generation

```python
from src.llm import answer_question

# No context (baseline)
answer = answer_question("What is the balcony like?")
print(answer)

# WITH context from RAG (recommended)
from src.rag import get_context_string

context = get_context_string("What is the balcony like?")
answer = answer_question("What is the balcony like?", context)
print(answer)
```

### Sample Questions to Test

```
1. "How big is the living room?"
2. "What color is the sofa?"
3. "Does the apartment have a balcony?"
4. "Tell me about the west-facing balcony"
5. "How many washrooms are there?"
6. "What's in the master bedroom?"
7. "Can I work from home here?"
8. "What can I see from the balcony?"
```

---

## 🚀 Deployment to Streamlit Cloud (FREE)

### Prerequisites
- GitHub account
- HuggingFace account with API token

### Step 1: Push to GitHub

```bash
# Initialize git (if not done)
git init
git add .
git commit -m "Initial commit: Home chatbot with RAG"

# Push to GitHub
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/Home-Chatbot.git
git push -u origin main
```

### Step 2: Deploy to Streamlit Cloud

1. Go to [Streamlit Cloud](https://share.streamlit.io)
2. Click **"New app"**
3. Select your GitHub repo: `Home-Chatbot`
4. Select branch: `main`
5. Select file: `app.py`
6. Click **"Deploy"**

### Step 3: Add Secrets

1. In Streamlit Cloud dashboard, click your app
2. Click **"Manage app"** → **"Secrets"**
3. Paste your secrets:
   ```toml
   HF_API_TOKEN = "hf_xxxxxxxxxxxxx"
   ```
4. App auto-reloads with secrets

**Your app is now live!** Share the URL with anyone.

---

## 🔧 Configuration Guide

### Change the LLM Model

Edit `src/config.py`:

```python
# Current (Mistral-7B - fast, free)
HF_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"

# Alternative options:
# HF_MODEL_ID = "meta-llama/Llama-2-7b-chat"  # More chatty
# HF_MODEL_ID = "tiiuae/falcon-7b-instruct"   # Different style
```

**Available free tier models:**
- Mistral-7B (recommended - fast)
- Llama-2-7b
- Falcon-7b
- Phi-2

### Change Embedding Model

Edit `src/config.py`:

```python
# Current (fast, 384-dim)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Alternative (more accurate, slower)
# EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
```

### Adjust RAG Parameters

Edit `src/config.py`:

```python
CHUNK_SIZE = 500          # Characters per chunk (smaller = more granular)
CHUNK_OVERLAP = 100       # Overlap between chunks
TOP_K_RETRIEVAL = 3       # Number of chunks to retrieve
```

---

## ❌ Troubleshooting

### "HF_API_TOKEN not found"
- Add token to `.streamlit/secrets.toml`
- Restart Streamlit app
- Verify token is valid at https://huggingface.co/settings/tokens

### "Model loading... Please try again"
- HuggingFace free tier can have slow cold starts
- Wait 30 seconds and retry
- Models are cached, second request will be faster

### "FAISS index not created"
- Ensure `docs/house_details.md` exists
- Check permissions in `vectorstore/` directory
- Delete `vectorstore/` folder and restart to rebuild

### Slow responses (~5-10 seconds per query)
- This is normal for free HuggingFace tier
- Upgrade to HuggingFace Pro for faster inference
- Or switch to local Ollama for faster responses

---

## 🎓 Understanding the RAG Pipeline

### How It Works

```
User Question: "How big is the living room?"
                          ↓
                [Convert to vector embedding]
                          ↓
                [Search FAISS index for 3 most similar chunks]
                          ↓
       Retrieved: "Living room dimensions: 8 feet wide by 14 feet long..."
                          ↓
       [Send question + context to LLM]
                          ↓
       Answer: "The living room is 8 feet wide by 14 feet long..."
```

### Components

1. **Document Loading** (`src/rag.py`)
   - Reads `docs/house_details.md`
   - Splits into chunks (200-500 chars each)

2. **Embeddings** (`src/rag.py`)
   - Converts each chunk to a 384-dimensional vector
   - Uses HuggingFace sentence-transformers

3. **FAISS Index** (`src/rag.py`)
   - Organizes vectors in a k-NN search structure
   - Enables similarity search in ~100ms

4. **Retrieval** (`src/rag.py`)
   - Finds top-k most similar chunks for user query
   - Returns context to LLM

5. **Generation** (`src/llm.py`)
   - Combines context + question in a prompt
   - Calls HuggingFace Inference API
   - Returns answer

---

## 📊 Performance Metrics

| Operation | Typical Time |
|-----------|--------------|
| FAISS Index Creation | 10-30s (one-time) |
| Chunk Retrieval | ~100ms |
| LLM Generation | 2-5s |
| Total Response | ~3-6s |

---

## 💰 Cost Breakdown (Monthly)

| Service | Cost | Notes |
|---------|------|-------|
| Streamlit Cloud | **FREE** | Unlimited apps, 1GB storage |
| HuggingFace Inference | **FREE** | Rate limit: ~15 req/min |
| FAISS (local) | **FREE** | Runs on Streamlit servers |
| Embeddings | **FREE** | Cached locally |
| **TOTAL** | **$0** | Completely free! |

---

## 🛣️ Future Enhancements

- [ ] Add image/photo gallery of apartment
- [ ] Add video tour integration
- [ ] Schedule a tour contact form
- [ ] Virtual 3D tour
- [ ] Multiple languages support
- [ ] User feedback/rating system
- [ ] Analytics dashboard
- [ ] Telegram bot integration

---

## 📝 License

This project is open source and available under the MIT License.

---

## 👤 Author

Created as a portfolio project to showcase RAG, LLM integration, and Streamlit deployment.

---

## 🤝 Support

For issues or questions:
1. Check the **Troubleshooting** section
2. Review HuggingFace API status
3. Check Streamlit logs in cloud dashboard

---

## 📚 Resources

- [Streamlit Docs](https://docs.streamlit.io)
- [LangChain Docs](https://python.langchain.com)
- [FAISS GitHub](https://github.com/facebookresearch/faiss)
- [HuggingFace Inference API](https://huggingface.co/inference-api)
- [Sentence Transformers](https://www.sbert.net)

---

**Happy Showcasing!** 🎉
