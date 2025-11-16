# core_rag.py
# Wrapper around RAG + LLM pipeline in BrainTumorChatbot/app_gradio_rag_only.py

from importlib.machinery import SourceFileLoader
from pathlib import Path
from typing import Dict
import sys

# --- Load app_gradio_rag_only directly from BrainTumorChatbot -----------------

PROJECT_DIR = Path(__file__).resolve().parent        # C:\Fatima_Final_Bot\MultimodalAssistant
ROOT_DIR = PROJECT_DIR.parent                   # C:\Fatima_Final_Bot
BRAIN_CHATBOT_DIR = ROOT_DIR / "BrainTumorChatbot"   # C:\Fatima_Final_Bot\BrainTumorChatbot

if not BRAIN_CHATBOT_DIR.is_dir():
    raise FileNotFoundError(
        f"Expected BrainTumorChatbot folder at {BRAIN_CHATBOT_DIR}, but it was not found."
    )

if str(BRAIN_CHATBOT_DIR) not in sys.path:
    sys.path.insert(0, str(BRAIN_CHATBOT_DIR))

APP_RAG_PATH = BRAIN_CHATBOT_DIR / "app_gradio_rag_only.py"

if not APP_RAG_PATH.is_file():
    raise FileNotFoundError(
        f"Expected app_gradio_rag_only.py at {APP_RAG_PATH}, but it was not found."
    )

_bt_rag_module = SourceFileLoader(
    "bt_app_gradio_rag_only", str(APP_RAG_PATH)
).load_module()

rag_answer = _bt_rag_module.answer  # function from the original app


# --- Audience prefixing --------------------------------------------------------

def _audience_prefix(audience: str) -> str:
    """
    Build an audience-specific instruction prefix that will be included
    in the question text passed to the RAG pipeline.
    """
    a = (audience or "").strip().lower()
    if a.startswith("clin"):
        # Clinician-facing
        return (
            "You are drafting a brief, clinically oriented explanation for a "
            "neurosurgeon or neuro-oncologist. Use correct medical terminology. "
            "Answer all parts of the question. Do not give patient-specific "
            "treatment decisions or drug doses.\n\n"
        )
    else:
        # Patient-facing by default
        return (
            "You are explaining this to a patient or family member with no "
            "medical background. Use clear, simple language and a reassuring tone. "
            "Answer every part of the question. Do NOT give a firm diagnosis or "
            "tell them exactly which treatment they must have; speak in general "
            "terms and encourage them to discuss details with their doctors.\n\n"
        )



def run_rag(question: str, audience: str = "patient", top_k: int = 5) -> Dict[str, str]:
    """
    Run the RAG + LLM pipeline with an audience-aware question.

    Parameters
    ----------
    question : str
        User question (plus any classifier summary we add upstream).
    audience : 'patient' or 'clinician'
    top_k : int
        Number of passages to retrieve.

    Returns
    -------
    dict with keys:
        - answer: str
        - sources_text: str
        - context_text: str
        - prompt_debug: str
        - raw_output: str
        - dbg: str
    """
    question = (question or "").strip()
    if not question:
        return {
            "answer": "",
            "sources_text": "",
            "context_text": "",
            "prompt_debug": "",
            "raw_output": "",
            "dbg": "Empty question.",
        }

    pref = _audience_prefix(audience)
    full_question = pref + question

    # BrainTumorChatbot.app_gradio_rag_only.answer
    text, sources, prompt, context_debug, raw_text, dbg = rag_answer(
        full_question, top_k
    )

    return {
        "answer": text,
        "sources_text": sources,
        "context_text": context_debug,
        "prompt_debug": prompt,
        "raw_output": raw_text,
        "dbg": dbg,
    }
