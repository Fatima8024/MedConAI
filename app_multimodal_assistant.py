# app_multimodal_assistant.py
# Unified Gradio interface: MRI image + question -> classifier + RAG answer

import gradio as gr

from core_classifier import run_classifier
from core_rag import run_rag

DISCLAIMER = (
    "⚠️ Educational use only. Not a medical device. "
    "Do NOT use this system for diagnosis or treatment decisions. "
    "Always consult a qualified clinician."
)

def multimodal_pipeline(image, question, audience, top_k):
    """
    Core logic:
    - If image provided: run classifier (+ overlay), build an image_summary using cls_status
    - Build a RAG question (image_summary + user question)
    - Run RAG + LLM
    """
    # 1) Classifier (if image provided)
    cls_label = "no_image"
    cls_conf = 0.0
    cls_expl = "No MRI image provided, classifier not run."
    overlay = None
    cls_status = "no_image"   # ✅ default, safe (no cls_result yet)

    label_map = {
    "glioma": "glioma",
    "meningioma": "meningioma",
    "pituitary": "pituitary tumour",
    "no_tumor": "no tumour detected",
    "no_tumour": "no tumour detected",
    }
    tumour_phrase = label_map.get(cls_label, cls_label)


    if image is not None:
        try:
         cls_result = run_classifier(image)
        except Exception as e:
            # ✅ prevent crash; show error in UI
         cls_result = {
            "adjusted_label": "unknown",
            "confidence": 0.0,
            "explanation": f"[Classifier ERROR] {type(e).__name__}: {e}",
            "overlay": None,
            "status": "error",
        }
 
        cls_result = run_classifier(image)
        cls_label = cls_result.get("adjusted_label", "unknown")
        cls_conf = float(cls_result.get("confidence", 0.0) or 0.0)
        cls_expl = cls_result.get("explanation", "(no explanation)")
        overlay = cls_result.get("overlay", None)

        # ✅ NEW: status-aware summary control (defaults to "confident" if missing)
        cls_status = cls_result.get("status", "confident")

    # 2) Build question for RAG
    user_q = (question or "").strip()

    # ---- TEXT-ONLY MODE ----
    if image is None:
        if user_q:
            rag_question = f"Question: {user_q}"
        else:
            rag_question = "Provide a general overview of brain tumours."

    # ---- IMAGE + CLASSIFIER MODE ----
    else:
        # safe readable tumour phrase (handles low_confidence_/uncertain_ prefixes if they still appear)
        tumour_phrase = (
            cls_label.replace("low_confidence_", "")
                    .replace("uncertain_", "")
                    .replace("_", " ")
        )

        # ✅ NEW: status-aware image summary (your snippet)
        if cls_status == "uncertain":
            image_summary = (
                f"MRI scan: classifier is uncertain (confidence {cls_conf:.2f}). "
                f"Top possibilities are close; consider alternatives."
            )
        elif cls_status == "low_confidence":
            image_summary = (
                f"MRI scan: classifier has low confidence (confidence {cls_conf:.2f}). "
                f"Treat this as provisional."
            )
        elif cls_label in ("no_tumor", "no_tumour"):
            image_summary = f"MRI scan: classifier detected no tumour (confidence {cls_conf:.2f})."
        else:
            image_summary = (
                f"MRI scan: classifier suggests {tumour_phrase} "
                f"(confidence {cls_conf:.2f}). Not confirmed diagnosis."
            )

        # Combine with user question (or default prompt)
        if user_q:
            rag_question = f"{image_summary}\n\nQuestion: {user_q}"
        else:
            rag_question = (
                f"{image_summary}\n\n"
                "Question: Explain what this could mean, where it occurs, symptoms, tests, and treatment options."
            )

    # 3) Run RAG + LLM
    rag_result = run_rag(
        question=rag_question,
        audience=audience,
        top_k=int(top_k) if top_k is not None else 5,
    )

    answer = rag_result.get("answer", "")
    sources_text = rag_result.get("sources_text", "")
    context_text = rag_result.get("context_text", "")
    dbg_text = rag_result.get("dbg", "")

    # 4) Return to UI
    return (
        cls_label,
        f"{cls_conf:.3f}",
        cls_status,
        cls_expl,
        overlay,
        answer,
        sources_text,
        context_text,
        dbg_text,
    )


def build_app():
    with gr.Blocks(title="Brain Tumor Multimodal Assistant") as demo:
        gr.Markdown("# 🧠 Brain Tumor Multimodal Assistant")
        gr.Markdown(DISCLAIMER)

        with gr.Row():
            with gr.Column(scale=1):
                image_in = gr.Image(label="MRI slice (optional)", type="pil")
                question_in = gr.Textbox(
                    label="Question (optional)",
                    lines=4,
                    placeholder="e.g. What does this kind of tumor usually mean? What are treatments?"
                )
                audience_in = gr.Dropdown(
                    ["Patient", "Clinician"],
                    value="Patient",
                    label="Audience mode"
                )
                topk_in = gr.Slider(
                    minimum=1,
                    maximum=5,
                    value=5,
                    step=1,
                    label="Top-K sources"
                )
                run_btn = gr.Button("Run")

            with gr.Column(scale=1):
                cls_label_out = gr.Textbox(label="Classifier label", interactive=False)
                cls_conf_out = gr.Textbox(label="Classifier confidence", interactive=False)
                cls_status_out = gr.Textbox(label="Classifier status", interactive=False)
                cls_expl_out = gr.Textbox(label="Classifier explanation", lines=10, interactive=False)
                overlay_out = gr.Image(label="Segmentation overlay (if available)")

        with gr.Row():
            answer_out = gr.Textbox(label="Answer", lines=8, interactive=False)
        with gr.Row():
            sources_out = gr.Textbox(label="Sources", lines=6, interactive=False)
            context_out = gr.Textbox(label="Context Used (Debug)", lines=10, interactive=False)
        with gr.Row():
            dbg_out = gr.Textbox(label="Debug", lines=3, interactive=False)

        run_btn.click(
            fn=multimodal_pipeline,
            inputs=[image_in, question_in, audience_in, topk_in],
            outputs=[
                cls_label_out,
                cls_conf_out,
                cls_status_out,
                cls_expl_out,
                overlay_out,
                answer_out,
                sources_out,
                context_out,
                dbg_out,
            ],
        )

    return demo


if __name__ == "__main__":
    demo = build_app()
    demo.launch(server_name="127.0.0.1", server_port=7861, show_error=True)
