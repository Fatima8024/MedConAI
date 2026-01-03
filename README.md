# MedConAI — Brain Tumor Multimodal Assistant (Applied ML + GenAI)

MedConAI is an educational multimodal assistant that combines:
1) an MRI tumor classifier (Applied ML), and  
2) a Retrieval-Augmented Generation (RAG) chatbot that produces document-grounded answers with citations (GenAI).

It supports two response modes:
- **Patient**: simpler, supportive explanations
- **Clinician**: more technical, structured explanations

> ⚠️ **Educational use only. Not a medical device.**  
> Do **not** use this system for diagnosis or treatment decisions. Always consult a qualified clinician.

---

## Demo (Screenshots)

**Clinician mode**  
![Clinician UI](assets/ui_clinician.png)

**Patient mode**  
![Patient UI](assets/ui_patient.png)

---

## Key features
- **Multimodal workflow:** MRI image input + (optional) segmentation overlay + text Q&A
- **Classifier output with confidence:** predicted tumor type + probability + explanation/debug panel
- **RAG with citations:** answers grounded in retrieved sources (shown in the UI)
- **Audience control:** patient vs clinician response style using the same evidence base
- **Safety behavior:** conservative language when evidence/confidence is weak + visible disclaimer

---

## Tech stack
- Python
- Gradio (UI)
- PyTorch (imaging model)
- LangChain + FAISS (retrieval)
- Azure OpenAI (chat completions) **or** local HF model fallback (FLAN-T5)
- dotenv for local configuration

---

## Repository files (what to look at)
- `app_multimodal_assistant.py` — main multimodal Gradio app (image + RAG)
- `app_gradio_rag_only.py` — RAG-only Gradio app (text chat + sources)
- `brain_tumor_corpus.py` — builds the RAG corpus + FAISS indexes (downloads sources)
- `core_classifier.py` — wrapper for classifier + overlay output
- `core_rag.py` — wrapper calling the RAG pipeline
- `build_eval_arrays.py`, `eval_classifier_metrics.py` — evaluation scripts
- `assets/` — screenshots / demo media

---

## Quickstart (local)

### 1) Create environment + install dependencies
```bash
python -m venv .venv

# Windows:
.venv\Scripts\activate

# Mac/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
