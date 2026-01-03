# core_rag.py
# Wrapper around RAG + LLM pipeline in BrainTumorChatbot/app_gradio_rag_only.py

from importlib.machinery import SourceFileLoader
from pathlib import Path
from typing import Dict
import sys

def _print_retrieval(docs_scores, used_docs, max_chars=400):
    print("\n" + "="*60)
    print("[RAG] RETRIEVED (top results before filtering):")
    for i, (d, sc) in enumerate(docs_scores[:10], 1):
        meta = d.metadata or {}
        src = meta.get("source_url") or meta.get("local_path") or meta.get("source") or "unknown"
        dt  = meta.get("doc_type") or "NONE"
        snippet = (d.page_content or "").replace("\n", " ")
        print(f"  ({i}) score={sc} | doc_type={dt} | src={src}")
        print(f"      {snippet[:max_chars]}{'...' if len(snippet) > max_chars else ''}")

    print("\n[RAG] USED IN PROMPT (after filtering/top_k):")
    for i, d in enumerate(used_docs, 1):
        meta = d.metadata or {}
        src = meta.get("source_url") or meta.get("local_path") or meta.get("source") or "unknown"
        dt  = meta.get("doc_type") or "NONE"
        snippet = (d.page_content or "").replace("\n", " ")
        print(f"  [{i}] doc_type={dt} | src={src}")
        print(f"      {snippet[:max_chars]}{'...' if len(snippet) > max_chars else ''}")
    print("="*60 + "\n")


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

_bt_rag_module._print_retrieval = _print_retrieval 

rag_answer = _bt_rag_module.answer  # function from the original app


# --- Audience prefixing --------------------------------------------------------

def _audience_prefix(audience: str) -> str:
    """
    Build a audience-specific prefix that  will be included
    in the question text passed to the RAG pipeline.
    The detailed behaviour, safety rules and style are handled in
    app_gradio_rag_only._build_prompt().
    """
    a = (audience or "").strip().lower()

    if a.startswith("clin"):
        # Clinician-facing
        return (
            # "This question is from a CLINICIAN "
            # "(e.g. radiologist, neurosurgeon, oncologist, or trainee). "
            # "They are comfortable with technical medical language and guideline-style discussion.\n\n"
            "This question is from a CLINICIAN (e.g. neurosurgeon, neuro-oncologist, "
            "neuroradiologist or trainee). Please give a concise but informative, "
            "clinically oriented explanation.\n\n"
            "- Use correct medical terminology and guideline-style language.\n"
            "- Start by briefly stating the MOST LIKELY tumour type and typical "
            "anatomical location, based on the documents and any information given.\n"
            "- Then summarise: usual clinical presentation / key symptoms, important "
            "imaging and pathology work-up, and standard management options "
            "(surgery, radiotherapy, systemic / endocrine treatments, monitoring).\n"
            "- You may mention relevant guideline-style phrases if the context supports "
            "them (for example WHO grade, IDH status, EANO / NCCN style wording).\n"
            "- Do NOT invent drug doses or give patient-specific treatment decisions; "
            "keep recommendations general and emphasise that final decisions are made "
            "by the treating team.\n\n"
        )
    else:
        # Patient-facing by default
        return (
            "This question is from a PATIENT or FAMILY MEMBER with no medical background. "
            "Please answer in clear, simple, reassuring language and explain medical terms in everyday words.\n\n"
            "- Begin with 1–2 sentences that gently explain what this tumour type is "
            "and where it usually occurs in the brain.\n"
            "- Then describe, in everyday words, the common symptoms people might "
            "notice and the usual tests doctors do (for example MRI or CT scans, "
            "blood tests, or biopsy).\n"
            "- Explain the typical treatment OPTIONS in general terms (such as "
            "surgery, radiotherapy, medicines or careful monitoring) without telling "
            "the person exactly which treatment they personally must have.\n"
            "- Avoid frightening language; be honest but balanced. Make it clear that "
            "any scanner / AI suggestion is only a POSSIBLE tumour type, not a final "
            "diagnosis, and that they must discuss details with their own doctors.\n\n"
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

    # NOTE: We NO LONGER prepend audience prefix here!
    # The audience parameter is now passed directly to the RAG function
    # so _build_prompt() can handle it properly
    
    # BrainTumorChatbot.app_gradio_rag_only.answer now takes audience parameter
    text, sources, prompt, context_debug, raw_text, dbg = rag_answer(
        question, top_k, audience  # <-- Pass audience as 3rd parameter
    )

    return {
        "answer": text,
        "sources_text": sources,
        "context_text": context_debug,
        "prompt_debug": prompt,
        "raw_output": raw_text,
        "dbg": dbg,
    }
