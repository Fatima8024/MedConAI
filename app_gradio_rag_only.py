# -*- coding: utf-8 -*-
import os, time, traceback
import gradio as gr
import re
import logging, warnings
from transformers.utils.logging import set_verbosity_error
set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)

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
RAG_DIR = os.getenv("RAG_DIR", os.path.join(BASE, "corpus", "rag_index"))

# LoRA finetuned adapters directory
LOCAL_FINETUNE_DIR = os.path.join(BASE, "models", "flan_t5_brain_qa")

# Base model
BASE_LLM_ID = "google/flan-t5-base"
DEFAULT_LLM_ID = "google/flan-t5-large"

# Embedding model
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# Generation / RAG params
MAX_NEW_TOK = 64
MAX_INPUT_TOK = 512
TOPK_DEFAULT = 3

GEN_KWARGS = dict(
    max_new_tokens=MAX_NEW_TOK,
    num_beams=6,
    do_sample=False,
    no_repeat_ngram_size=2,
    repetition_penalty=1.2,
    early_stopping=True,
)

# ================== GLOBALS ==================
_emb = None
_vs = None
_llm = None
_tok = None

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
    """
    Load answer LLM.
    """
    global _llm, _tok
    if _llm is not None:
        return _llm
    
    use_lora = False

    # use_lora = (
    #     os.path.isdir(LOCAL_FINETUNE_DIR)
    #     and os.path.exists(os.path.join(LOCAL_FINETUNE_DIR, "adapter_config.json"))
    #     and os.path.exists(os.path.join(LOCAL_FINETUNE_DIR, "adapter_model.safetensors"))
    #     and PEFT_AVAILABLE
    # )

    if use_lora:
        print(f"[LLM] Loading base {BASE_LLM_ID} with LoRA from {LOCAL_FINETUNE_DIR}")
        base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_LLM_ID)
        lora_model = PeftModel.from_pretrained(base_model, LOCAL_FINETUNE_DIR)
        model = lora_model.merge_and_unload()
        _tok = AutoTokenizer.from_pretrained(BASE_LLM_ID, use_fast=True)
    else:
        target = os.getenv("LLM_MODEL_ID", DEFAULT_LLM_ID)
        print(f"[LLM] Loading base model without LoRA: {target}")
        model = AutoModelForSeq2SeqLM.from_pretrained(target)
        _tok = AutoTokenizer.from_pretrained(target, use_fast=True)
    # Set tokenizer limits
    _tok.model_max_length = MAX_INPUT_TOK
    _tok.truncation_side = "left"

    _llm = Text2TextGenerationPipeline(
        model=model,
        tokenizer=_tok,
        device=-1,  # CPU
        **GEN_KWARGS,
    )
    print(f"[LLM] Device set to CPU, max input tokens: {MAX_INPUT_TOK}")
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


def _is_qa_training_data(content: str) -> bool:
    """Return True only for obvious multi-Q/A training shards - higher threshold."""
    if not content:
        return False
    t = content.lower()

    # Count explicit Q/A labels - require more for flagging
    q_count = t.count("q:") + t.count("question:")
    a_count = t.count("a:") + t.count("answer:")

    if (q_count + a_count) >= 4:  # Increased from 3
        return True

    question_marks = t.count("?")
    # bullet_count = content.count("•") + content.count("-")
    # if bullet_count >= 15 and question_marks >= 8:  # Increased thresholds
    #     return True

    # Count real bullets at line starts, not hyphens inside words
    bullet_count = len(re.findall(r'(?m)^\s*[•-]\s+', content))
    #question_marks = t.count("?")

    # Only treat as QA if it's BOTH bullet-heavy AND question-heavy
    return bullet_count >= 8 and question_marks >= 4
    #return False


def _build_prompt(question: str, docs, max_input_tokens: int = MAX_INPUT_TOK) -> str:
    """
    Build a clean, focused QA prompt that fits within token limits
    """
    instruction = (
    "Imagine you are a brain tumor specialist and have knowledge about it. You know all the terminologies, what tests, scans, treatments are required,you know the symptoms and cure. Based on the medical context below and your knowledge, provide a logical, understandable answer in 1–3 sentences from the medical context. "
    "Do not invent information. Do not include 'Q:' or 'A:' or bullet lists. "
    #"If the information is not in the context, say \"I cannot find specific information about this in the available documents.\"\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
    )


    # Token budget for context
    base_prompt = instruction.format(context="", question=question.strip())
    base_tokens = _count_tokens(base_prompt)
    available_tokens = max_input_tokens - base_tokens - 20
    if available_tokens < 50:
        available_tokens = 50

    # Build context from cleaned docs
    context_parts = []
    current_tokens = 0
    for i, d in enumerate(docs):
        raw_content = (d.page_content or "").strip()
        if _is_qa_training_data(raw_content):
            continue

    
        clean_content = _clean_content(raw_content)
        if not clean_content:
            continue

        content_tokens = _count_tokens(clean_content)
        if current_tokens + content_tokens > available_tokens:
            remaining = available_tokens - current_tokens
            if remaining > 20:
                trunc = _truncate_to_tokens(clean_content, remaining)
                if trunc:
                    context_parts.append(f"[{i+1}] {trunc}")
            break

        context_parts.append(f"[{i+1}] {clean_content}")
        current_tokens += content_tokens

    if not context_parts:
        # Fall back to longer raw snippets
        avail = max_input_tokens - _count_tokens(instruction.format(context="", question=question.strip())) - 20
        per_doc = max(32, avail // max(1, len(docs)))  # Increased min from 32
        for i, d in enumerate(docs):
            raw = (d.page_content or "").strip()
            if not raw:
                continue
            snippet = _truncate_to_tokens(raw, per_doc)
            if snippet:
                context_parts.append(f"[{i+1}] {snippet}")

    context = "\n\n".join(context_parts) if context_parts else "No relevant context found."
    full_prompt = instruction.format(context=context, question=question.strip())
    return _truncate_to_tokens(full_prompt, max_input_tokens)

# def _extract_answer(text: str) -> str:
#     DEFAULT = "I cannot find specific information about this in the available documents."
#     if not text:
#         return DEFAULT

#     m = re.search(r'(?:^|\n)(?:answer:|a:)\s*(.*)$', text, flags=re.IGNORECASE | re.DOTALL)
#     if m:
#         text = m.group(1).strip()

#     # Strip prompt echoes
#     text = re.sub(r'(?im)^(?:context|question)\s*:\s*.*$', '', text).strip()

#     # Take the first 1–3 sentences that look declarative
#     sentences = re.split(r'(?<=[.!?])\s+', text)
#     picked = []
#     for s in sentences:
#         s = s.strip()
#         if len(s) >= 8 and not s.lower().startswith(("context:", "question:", "based on", "according to")):
#             picked.append(s)
#         if len(picked) >= 3:
#             break

#     return (" ".join(picked)).strip() or DEFAULT

def _extract_answer(text: str) -> str:
    if not text:
        return "I cannot find specific information about this in the available documents."

    # --- keep only what comes after the LAST "Answer:" or "A:" (case-insensitive) ---
    m_last = None
    for m in re.finditer(r'(?:^|\n)\s*(?:answer|a)\s*:\s*(.*)$', text, flags=re.IGNORECASE | re.DOTALL):
        m_last = m
    if m_last:
        text = m_last.group(1).strip()

    # --- very light cleanup: only drop bullets that are questions ---
    kept = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(('•', '-')) and line.rstrip().endswith('?'):
            continue
        kept.append(line)

    clean = ' '.join(kept).strip()
    if not clean:
        return "I cannot find specific information about this in the available documents."

    # --- return the first 1–2 sentences ---
    sentences = re.split(r'(?<=[.!?])\s+', clean)
    return ' '.join(sentences[:2]).strip()


# ================== CORE ==================filtered_docs:

def on_load(rag_dir):
    try:
        return _load_index(rag_dir)
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


def answer(question, top_k):
    question = (question or "").strip()
    if not question:
        return "", "", "", "", "", "Please type a question."

    try:
        global _vs

        # Ensure index + LLM are ready
        status = None
        if _vs is None:
            status = _load_index(RAG_DIR)
        _ = _load_llm()  # ensures global _llm/_tok are initialized

        # Retrieval
        k = min(int(top_k), 5) if top_k else TOPK_DEFAULT
        filtered_docs = []
        docs_scores = []
        mmr_docs = []

        try:
            mmr_docs = _vs.max_marginal_relevance_search(
                question, k=k, fetch_k=32, lambda_mult=0.3
            )
            docs_scores = _vs.similarity_search_with_score(question, k=max(k * 6, 24))
        except Exception:
            try:
                docs_scores = _vs.similarity_search_with_score(question, k=max(k * 6, 24))
            except TypeError:
                base_docs = _vs.similarity_search(question, k=max(k * 6, 24))
                docs_scores = [(d, None) for d in base_docs]

        # Guard: nothing retrieved at all
        if not docs_scores and not mmr_docs:
            return (
                "I cannot find specific information about this in the available medical documents.",
                "", "", "", "",
                "Retriever returned 0 passages."
            )

        # Filter documents
        for doc, score in docs_scores:
            content = (doc.page_content or "").strip()
            meta = doc.metadata or {}

            source = (meta.get("source") or "").lower()
            if "qa_jsonl" in source or "qa_json" in source:
                continue
            if _is_qa_training_data(content):
                continue
            if len(content) < 25:
                continue

            filtered_docs.append(doc)
            if len(filtered_docs) >= k:
                break

        # Fallbacks
        if not filtered_docs and docs_scores:
            filtered_docs = [d for d, _ in docs_scores[:max(1, k)]]
        if not filtered_docs and mmr_docs:
            filtered_docs = mmr_docs[:max(1, k)]
        if not filtered_docs:
            return (
                "I cannot find specific information about this in the available medical documents.",
                "", "", "", "",
                "No relevant passages after filtering."
            )

        # Build prompt and generate
        prompt = _build_prompt(question, filtered_docs)
        inputs = _llm.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_INPUT_TOK,
            add_special_tokens=True,
        )
        outputs = _llm.model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOK,
            num_beams=4,
            do_sample=False,
            no_repeat_ngram_size=3,
            repetition_penalty=1.1,
            early_stopping=True,
        )
        raw_text = _llm.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        # Keep only content after "Answer:" or "A:" if present
        m = re.search(r'(?:^|\n)(?:answer|a)\s*:\s*(.*)$', raw_text,
                      flags=re.IGNORECASE | re.DOTALL)
        if m:
            raw_text = m.group(1).strip()

        text = _extract_answer(raw_text)

        # Fallback: if the model hedged but the context plainly contains statements,
        # use the first 1–2 sentences from the context we actually sent.
        if not text or text.lower().startswith("i cannot find"):
         # the same slice you show in "Context Used (Debug)"
            context_debug = prompt.split("Context:")[1].split("Question:")[0].strip() if "Context:" in prompt else ""
            if context_debug:
                    ctx_sents = re.split(r'(?<=[.!?])\s+', context_debug)
                    ctx_pick = [s.strip() for s in ctx_sents if len(s.strip()) > 20][:2]
                    if ctx_pick:
                        text = " ".join(ctx_pick)


        # Sources
        src_lines = []
        for i, d in enumerate(filtered_docs, 1):
            meta = d.metadata or {}
            src = (meta.get("source_url") or meta.get("local_path")
                   or meta.get("source") or "")
            if src and "qa_jsonl" not in src.lower() and "qa_json" not in src.lower():
                src_lines.append(f"[{i}] {src}")
        sources = "\n".join(src_lines) if src_lines else "(source information not available)"

        # Debug fields
        context_debug = (
            prompt.split("Context:")[1].split("Question:")[0].strip()
            if "Context:" in prompt else "No context"
        )
        prompt_tokens = _count_tokens(prompt)
        dbg = status or f"OK | used {len(filtered_docs)} passages | prompt tokens: {prompt_tokens}/{MAX_INPUT_TOK}"

        return text, sources, prompt, context_debug, raw_text, dbg

    except Exception as e:
        return "", "", "", "", "", f"ERROR: {type(e).__name__}: {e}\n{traceback.format_exc()}"

# ================== UI ==================

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
    topk = gr.Slider(1, 5, value=TOPK_DEFAULT, step=1, label="Top-K passages")
    ask_btn = gr.Button("Answer", variant="primary")

    ans = gr.Textbox(label="Answer", lines=10)
    src = gr.Textbox(label="Sources", lines=6)
    prompt_debug = gr.Textbox(label="Full Prompt (Debug)", lines=8)  # New
    context_debug = gr.Textbox(label="Context Used (Debug)", lines=6)  # New
    raw_output = gr.Textbox(label="Raw Model Output (Debug)", lines=4)  # New
    dbg = gr.Textbox(label="Debug", lines=4)

    load_btn.click(on_load, inputs=[rag_dir_in], outputs=status_box)
    ask_btn.click(answer, inputs=[q, topk], outputs=[ans, src, prompt_debug, context_debug, raw_output, dbg])  # Updated outputs

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7861, show_error=True)