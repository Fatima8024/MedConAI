# -*- coding: utf-8 -*-
import os, time, traceback
import gradio as gr
import re
import logging, warnings
from transformers.utils.logging import set_verbosity_error
set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
from openai import AzureOpenAI
from dotenv import load_dotenv
load_dotenv() 
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")


from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Text2TextGenerationPipeline,
)

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

# ================== CONFIG ==================

BASE = os.path.dirname(os.path.abspath(__file__))

# FAISS index dir (built by brain_tumor_corpus.py)
RAG_DIR = os.getenv("RAG_DIR", os.path.join(BASE, "corpus", "rag_index_evidence"))

#RAG_DIR = "C:\Fatima_Final_Bot\BrainTumorChatbot\corpus\rag_index_evidence"

# LoRA finetuned adapters directory
LOCAL_FINETUNE_DIR = os.path.join(BASE, "models", "flan_t5_brain_qa")

# Base model
BASE_LLM_ID = "google/flan-t5-base"
DEFAULT_LLM_ID = "google/flan-t5-large"

# Embedding model
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT", "https://fatimagpt.cognitiveservices.azure.com/")
AZURE_API_KEY = os.getenv("AZURE_API_KEY", "")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-12-01-preview")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME", "o1")

LLM_MODE = os.getenv("LLM_MODE", "azure")
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "2000"))



# Token limits

MAX_DOCS_FOR_CONTEXT = 4
MAX_CHARS_PER_DOC = 1200

# Legacy settings (for local model fallback)
# Generation 
MAX_NEW_TOK = 384
MAX_INPUT_TOK = 512
TOPK_DEFAULT = 3

GEN_KWARGS = dict(
    max_new_tokens=MAX_NEW_TOK,
    num_beams=4,
    do_sample=True,
    no_repeat_ngram_size=3,
    repetition_penalty=1.1,
    #early_stopping=True,
    temperature=0.7, 
    top_p=0.9
)

# ================== GLOBALS ==================
_emb = None
_vs = None
_llm = None
_tok = None
_azure_client = None

# STEP 3: Add these NEW FUNCTIONS for Azure OpenAI
# ============================================================================

def _init_azure():
    """Initialize Azure OpenAI client"""
    global _azure_client
    if _azure_client is None:
        if not AZURE_API_KEY or AZURE_API_KEY == "<your-api-key>":
            raise ValueError(
                "Azure API key not set! Please set AZURE_API_KEY in the config section."
            )
        
        _azure_client = AzureOpenAI(
            api_version=AZURE_API_VERSION,
            azure_endpoint=AZURE_ENDPOINT,
            api_key=AZURE_API_KEY,
        )
        print(f"[LLM] ✓ Azure OpenAI initialized successfully")
        print(f"[LLM] Endpoint: {AZURE_ENDPOINT}")
        print(f"[LLM] Deployment: {AZURE_DEPLOYMENT_NAME}")
    
    return _azure_client


def _build_prompt_azure(question: str, docs, audience: str = "Patient") -> tuple:
    """Build system and user prompts for Azure OpenAI."""
    
    is_clinician = (audience or "").strip().lower().startswith("clin")
    
    if is_clinician:
        system_prompt = """Medical AI for healthcare professionals. Provide evidence-based responses with medical terminology covering: tumor type/location, symptoms, diagnostics, treatments. Keep general; specifics determined by treating team."""
    else:
        system_prompt = """Medical AI helping patients understand brain tumors. Use simple language. Cover: what tumor is, where it grows, symptoms, tests, treatments. Clarify AI predictions aren't final diagnoses."""

    # Build context
    docs = list(docs)[:MAX_DOCS_FOR_CONTEXT]
    context_parts = []
    
    for i, doc in enumerate(docs, 1):
        content = (doc.page_content or "").strip()
        if content:
            truncated = content[:MAX_CHARS_PER_DOC]
            context_parts.append(f"[{i}] {truncated}")
    
    context = "\n\n".join(context_parts) if context_parts else "No relevant information found."
    
    user_prompt = f"""Context: {context}

Q: {question}

Provide a complete answer (6-8 sentences)."""

    return system_prompt, user_prompt


def _call_azure(system_prompt: str, user_prompt: str, audience: str) -> str:
    """Call Azure OpenAI API."""
    client = _init_azure()
    
    try:
        response = client.chat.completions.create(
            model=AZURE_DEPLOYMENT_NAME,  # ← Uses your deployment name
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            reasoning_effort="low", 
            max_completion_tokens=MAX_OUTPUT_TOKENS,  # ← Note: max_completion_tokens for Azure
        )
        
        # Log usage
        usage = response.usage
        print(f"[AZURE] Tokens - Prompt: {usage.prompt_tokens}, Completion: {usage.completion_tokens}, Total: {usage.total_tokens}")
        
        # Azure o1 pricing (adjust if using different model)
        # o1 is expensive: ~$15/1M prompt tokens, ~$60/1M completion tokens
        cost = (usage.prompt_tokens * 15 + usage.completion_tokens * 60) / 1_000_000
        print(f"[AZURE] Estimated cost: ${cost:.6f}")
        
        text = response.choices[0].message.content
        text = (text or "").strip()

        if not text:
            # fall back so UI never stays blank
            text = (
                "I couldn't generate a complete answer within the current token limit. "
                "Please try again, or increase MAX_OUTPUT_TOKENS."
            )

        return text

               
    except Exception as e:
        print(f"[AZURE ERROR] {type(e).__name__}: {e}")
        return f"Azure OpenAI API Error: {str(e)}\nPlease check your API key and endpoint."


# ================== HELPERS ==================

def _init_embeddings():
    global _emb
    if _emb is None:
        _emb = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _emb


def _load_index(rag_dir: str):
    global _vs

    print("[RAG] Loading index from:", rag_dir)  
    emb = _init_embeddings()
    faiss_path = os.path.join(rag_dir, "index.faiss")
    pkl_path = os.path.join(rag_dir, "index.pkl")
    if not (os.path.exists(faiss_path) and os.path.exists(pkl_path)):
        raise FileNotFoundError(
            f"Missing index files in: {rag_dir}\n"
            "Run brain_tumor_corpus.py first to build the index."
        )
    _vs = FAISS.load_local(rag_dir, emb, allow_dangerous_deserialization=True)
    m1 = time.ctime(os.path.getmtime(faiss_path))
    m2 = time.ctime(os.path.getmtime(pkl_path))
    return (
        f"Loaded index (ntotal={_vs.index.ntotal}, dim={_vs.index.d})\n"
        f"index.faiss: {m1}\nindex.pkl:   {m2}"

    )


def _load_llm():
    """Load LLM - Azure OpenAI or local model based on LLM_MODE."""
    global _llm, _tok
    
    if LLM_MODE == "azure":
        print("[LLM] Mode: Azure OpenAI")
        _init_azure()
        return None
    
    # Local model fallback (same as before)
    if _llm is not None:
        return _llm
    
    print("[LLM] Mode: Local model (LoRA or base)")
    
    try:
        from peft import PeftModel
        PEFT_AVAILABLE = True
    except ImportError:
        PEFT_AVAILABLE = False
    
    use_lora = (
        LLM_MODE == "lora"
        and PEFT_AVAILABLE
        and os.path.isdir(LOCAL_FINETUNE_DIR)
        and os.path.exists(os.path.join(LOCAL_FINETUNE_DIR, "adapter_config.json"))
    )

    if use_lora:
        print(f"[LLM] Loading LoRA model from {LOCAL_FINETUNE_DIR}")
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Text2TextGenerationPipeline
        base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_LLM_ID)
        lora_model = PeftModel.from_pretrained(base_model, LOCAL_FINETUNE_DIR)
        model = lora_model.merge_and_unload()
        _tok = AutoTokenizer.from_pretrained(BASE_LLM_ID, use_fast=True)
    else:
        print(f"[LLM] Loading base model: {DEFAULT_LLM_ID}")
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Text2TextGenerationPipeline
        model = AutoModelForSeq2SeqLM.from_pretrained(DEFAULT_LLM_ID)
        _tok = AutoTokenizer.from_pretrained(DEFAULT_LLM_ID, use_fast=True)
    
    _tok.model_max_length = 512
    _tok.truncation_side = "left"
    _llm = Text2TextGenerationPipeline(model=model, tokenizer=_tok, device=-1, **GEN_KWARGS)
    
    return _llm


def _count_tokens(text: str) -> int:
    """Count tokens in text"""
    global _tok
    if _tok is None:
        _load_llm()
    return len(_tok.encode(text, truncation=False, add_special_tokens=False))


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to max tokens"""
    global _tok
    if _tok is None:
        _load_llm()
    
    tokens = _tok.encode(text, truncation=False, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return text
    
    truncated_tokens = tokens[:max_tokens]
    return _tok.decode(truncated_tokens, skip_special_tokens=True)


def _clean_content(content: str) -> str:
    """Clean document content to remove QA formatting and other noise - less aggressive"""
    if not content:
        return ""
    
    content = re.sub(r'Q:\s.*?\bA:\s*', '', content, flags=re.IGNORECASE | re.DOTALL)

    
    # Remove QA pairs that look like training data
    lines = content.split('\n')
    clean_lines = []
    
    for line in lines:
        line = line.strip()
        if not line or len(line) < 5:
            continue
        
        m_q = re.match(r'^(?:Q:|Question:)\s*(.*)$', line, flags=re.IGNORECASE)
        if m_q:
            qbody = m_q.group(1).strip()
            if qbody.endswith("?"):
                continue  # real question line → skip
            line = qbody  # keep if it looks like a statement

        # If the line starts with A:/Answer:, KEEP the content (do not skip!)
        m_a = re.match(r'^(?:A:|Answer:)\s*(.*)$', line, flags=re.IGNORECASE)
        if m_a:
            line = m_a.group(1).strip()
        clean_lines.append(line)
    
    clean_content = " ".join(clean_lines)
    
    # Remove multiple spaces
    clean_content = re.sub(r"\s+", " ", clean_content).strip()
    
    return clean_content


# def _is_qa_training_data(content: str) -> bool:
#     """Return True only for obvious multi-Q/A training shards - higher threshold."""
#     if not content:
#         return False
#     t = content.lower()

#     # Count explicit Q/A labels - require more for flagging
#     q_count = t.count("q:") + t.count("question:")
#     a_count = t.count("a:") + t.count("answer:")

#     if (q_count + a_count) >= 10:  # Increased from 3
#         return True
#     bullet_count = len(re.findall(r'(?m)^\s*[•-]\s+', content))
#     question_marks = t.count("?")
#     return bullet_count >= 20 and question_marks >= 10
#     # bullet_count = content.count("•") + content.count("-")
#     # if bullet_count >= 15 and question_marks >= 8:  # Increased thresholds
#     #     return True

#     # Count real bullets at line starts, not hyphens inside words
    
#     #question_marks = t.count("?")

#     # Only treat as QA if it's BOTH bullet-heavy AND question-heavy
    
#     #return False

def _build_prompt(question: str, docs, audience: str = "Patient", max_input_tokens: int = MAX_INPUT_TOK) -> str:
    """
    Build a very compact RAG prompt that fits in 512 tokens.
    """
    
    is_clinician = (audience or "").strip().lower().startswith("clin")
    
    # VERY SHORT instructions
    if is_clinician:
        instruction = """Using CONTEXT below, answer clinically: (1) tumor type & location, (2) symptoms, (3) tests needed, (4) treatment options. Use medical terms. 6-8 sentences."""
    else:
        instruction = """Using CONTEXT below, answer in simple words: (1) what this tumor is & where it grows, (2) symptoms, (3) tests doctors do, (4) treatments available. Explain terms simply. 6-8 sentences."""

    # Limit to 3 docs with smaller budget per doc
    docs = list(docs)[:3]
    context_lines = []
    per_doc_budget = 70  # Reduced from 90

    for i, doc in enumerate(docs, 1):
        content = (doc.page_content or "").strip()
        if not content:
            continue
        snippet = _truncate_to_tokens(content, per_doc_budget)
        if snippet:
            context_lines.append(f"[{i}] {snippet}")

    context = "\n".join(context_lines) if context_lines else "No context."

    # Ultra-compact structure
    full_prompt = f"""{instruction}

CONTEXT:
{context}

Q: {question.strip()}

A:"""

    total_tokens = _count_tokens(full_prompt)
    print(f"[RAG DEBUG] Audience={audience} | Prompt tokens={total_tokens}/{max_input_tokens}")

    return full_prompt


def _extract_answer(text: str) -> str:
    """
    Extract clean answer from LLM output.
    Handles prompt echoes, partial outputs, and formatting issues.
    """
    DEFAULT = "I cannot find specific information about this in the available documents."

    text = (text or "").strip()
    if not text:
        return DEFAULT

    # If output starts with "A:" or "ANSWER:", take what comes after
    if text.startswith(("A:", "A :", "ANSWER:", "Answer:")):
        text = re.sub(r'^A\s*:\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^ANSWER\s*:\s*', '', text, flags=re.IGNORECASE)
        text = text.strip()

    # Remove any context/question echoes
    text = re.sub(r'(?i)^.*?(?:CONTEXT|QUESTION|Q)\s*:.*?(?=\n|$)', '', text, flags=re.MULTILINE)
    
    # Remove instruction echoes
    instruction_patterns = [
        r'(?i)using context below.*?(?:\.|$)',
        r'(?i)answer (?:in simple words|clinically).*?(?:\.|$)',
        r'(?i)MRI scan:.*?Not confirmed diagnosis\.',
    ]
    
    for pattern in instruction_patterns:
        text = re.sub(pattern, '', text)
    
    # Clean up
    text = re.sub(r'\s+', ' ', text).strip()
    
    # If it's just the scan description echoed back, return default
    if len(text) < 40 or text.lower().startswith('mri scan:'):
        return DEFAULT
    
    # Remove any remaining formatting artifacts
    text = text.replace('**', '').strip()
    
    return text or DEFAULT


# ================== CORE ==================filtered_docs:

def on_load(rag_dir):
    try:
        return _load_index(rag_dir)
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


# STEP 4: REPLACE the answer() function with this Azure-aware version
# ============================================================================

def answer(question, top_k, audience="Patient"):
    """Main RAG answer function - uses Azure OpenAI or local model based on LLM_MODE."""

    question = (question or "").strip()
    if not question:
        return "", "", "", "", "", "Please type a question."

    try:
        global _vs

        if _vs is None:
            status = _load_index(RAG_DIR)

        _ = _load_llm()

        # Retrieval
        k = min(int(top_k), 5) if top_k else TOPK_DEFAULT
        filtered_docs = []

        try:
            docs_scores = _vs.similarity_search_with_score(question, k=max(k * 4, 20))
        except TypeError:
            base_docs = _vs.similarity_search(question, k=max(k * 4, 20))
            docs_scores = [(d, None) for d in base_docs]

        from collections import Counter
        print("DOC TYPE COUNTS:", Counter(((d.metadata or {}).get("doc_type", "NONE") for d, _ in docs_scores)))

        if not docs_scores:
            return ("No relevant information found.", "", "", "", "", "No retrieval results.")

        # Filter (keep only evidence docs, skip qa_pair + unknown sources)
        for doc, score in docs_scores:
            content = (doc.page_content or "").strip()
            meta = doc.metadata or {}

            doc_type = (meta.get("doc_type") or "").lower()
            source = (meta.get("source") or "").lower()
            qa_source = (meta.get("qa_source") or "").lower()

            # Skip synthetic QA pairs
            if doc_type == "qa_pair":
                continue

            # Skip anything with unknown/empty source
            if source in ("", "unknown"):
                continue

            # Old filter (optional)
            if "qa_jsonl" in source or "qa_json" in source or "qa_jsonl" in qa_source:
                continue

            filtered_docs.append(doc)
            if len(filtered_docs) >= k:
                break

        # IMPORTANT: do NOT fall back to QA silently
        if not filtered_docs:
            return (
                "No evidence documents found in retrieval (only qa_pair chunks were returned).",
                "", "", "", "",
                "No non-QA sources available. Rebuild/load the evidence index."
            )

        # ✅ ADD THIS BLOCK HERE (RAG vs USED)
        used_docs = filtered_docs[:k]
        try:
            _print_retrieval(docs_scores, used_docs)
        except Exception:
            pass

        print(f"[RAG] Retrieved {len(docs_scores)} docs, using {len(filtered_docs)} | Mode: {LLM_MODE} | Audience: {audience}")
        print("META SAMPLE:", (filtered_docs[0].metadata if filtered_docs else None))

        # Generation
        if LLM_MODE == "azure":
            system_prompt, user_prompt = _build_prompt_azure(question, filtered_docs, audience)

            raw_text = _call_azure(system_prompt, user_prompt, audience)  # raw from model
            text = _extract_answer(raw_text)  # cleaned + default if empty

            prompt_debug = f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}"

            print(f"\n{'='*60}")
            print(f"AZURE ANSWER ({audience}) RAW:\n{raw_text}")
            print(f"\nCLEANED:\n{text}")
            print('='*60)

        else:
            # Local model path
            prompt_debug = _build_prompt_local(question, filtered_docs, audience)
            result = _llm(prompt_debug, max_new_tokens=200, min_length=60)
            raw_text = result[0]['generated_text'] if result else ""
            text = _extract_answer(raw_text)

        # Sources
        src_lines = []
        for i, d in enumerate(filtered_docs, 1):
            meta = d.metadata or {}
            src = meta.get("source_url") or meta.get("local_path") or meta.get("source") or "unknown"
            if "qa_jsonl" not in src.lower():
                src_lines.append(f"[{i}] {src}")

        sources = "\n".join(src_lines) if src_lines else "(no sources)"
        context_debug = "\n\n".join([f"[{i}] {d.page_content[:1500]}..." for i, d in enumerate(filtered_docs, 1)])
        dbg = f"OK | Mode: {LLM_MODE} | {len(filtered_docs)} docs | Audience: {audience}"

        return text, sources, prompt_debug, context_debug, raw_text, dbg

    except Exception as e:
        import traceback
        return ("", "", "", "", "", f"ERROR: {type(e).__name__}: {e}\n{traceback.format_exc()}")

# ================== UI ==================
# Replace the existing Gradio UI section in app_gradio_rag_only.py

with gr.Blocks(title="Brain Tumor RAG — Medical QA") as demo:
    gr.Markdown("## 🧠 Brain Tumor RAG — Medical QA")

    rag_dir_in = gr.Textbox(value=RAG_DIR, label="RAG_DIR")
    load_btn = gr.Button("Load Index", variant="primary")
    status_box = gr.Textbox(label="Status", value="(click Load Index)", lines=4)

    q = gr.Textbox(
        label="Ask a question",
        lines=3,
        placeholder="e.g., What are the treatment options for brain tumors? What are the symptoms of glioma?",
    )
    
    # Add audience selector
    audience_selector = gr.Radio(
        ["Patient", "Clinician"],
        value="Patient",
        label="Audience (who is asking?)"
    )
    
    topk = gr.Slider(1, 5, value=TOPK_DEFAULT, step=1, label="Top-K passages")
    ask_btn = gr.Button("Answer", variant="primary")

    ans = gr.Textbox(label="Answer", lines=10)
    src = gr.Textbox(label="Sources", lines=6)
    prompt_debug = gr.Textbox(label="Full Prompt (Debug)", lines=8)
    context_debug = gr.Textbox(label="Context Used (Debug)", lines=16)
    raw_output = gr.Textbox(label="Raw Model Output (Debug)", lines=4)
    dbg = gr.Textbox(label="Debug", lines=4)

    load_btn.click(on_load, inputs=[rag_dir_in], outputs=status_box)
    
    # Update to include audience parameter
    ask_btn.click(
        answer, 
        inputs=[q, topk, audience_selector],  # <-- Added audience_selector
        outputs=[ans, src, prompt_debug, context_debug, raw_output, dbg]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7861, show_error=True)
