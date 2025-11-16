# app_multimodal_assistant.py
# Unified Gradio interface: MRI image + question -> classifier + RAG answer

import gradio as gr

from core_classifier import run_classifier
from core_rag import run_rag


DISCLAIMER = (
    "⚠️ Educational use only. Not a medical device. "
    "Do NOT use this system for diagnosis or treatment decisions. "
    "Always consult qualified clinicians for medical care."
)


def multimodal_pipeline(image, question, audience, top_k):
    """
    Core logic for the assistant:
    - Optionally run classifier on MRI image
    - Build a combined question for RAG
    - Run RAG with audience-aware instructions
    - Return classifier + RAG outputs for the UI
    """
    # 1) Classifier (if image provided)
    cls_label = "no_image"
    cls_conf = 0.0
    cls_expl = "No MRI image provided, classifier not run."
    overlay = None

    if image is not None:
        cls_result = run_classifier(image)
        cls_label = cls_result.get("adjusted_label", "unknown")
        cls_conf = float(cls_result.get("confidence", 0.0) or 0.0)
        cls_expl = cls_result.get("explanation", "(no explanation)")
        overlay = cls_result.get("overlay", None)

    # 2) Build combined question string for RAG
    user_q = (question or "").strip()

    if image is None:
        # ---- TEXT-ONLY MODE: ignore classifier, just answer the question ----
        if user_q:
            rag_question = (
                "The user has asked a general question about brain tumors.\n\n"
                f"Question: {user_q}\n\n"
                "Using only the provided medical documents, please answer this question. "
                "If the question has several parts, make sure you answer each part. "
                "Do NOT give a diagnosis or a specific personal treatment decision; "
                "encourage them to discuss options with their own doctors."
            )
        else:
            rag_question = (
                "Provide a general educational overview about brain tumors using the "
                "provided medical documents only. Do not make any treatment decisions."
            )

    else:
        # ---- IMAGE + CLASSIFIER MODE ----
        if cls_label == "no_tumor":
            image_summary = (
                f"The computer model does not see a clear tumor in this slice "
                f"(predicted class: 'no_tumor', confidence ~{cls_conf:.2f}). "
                "This can be wrong; human radiologist review is essential."
            )
        else:
            image_summary = (
                "The model highlights a region in red as suspicious. Its current best "
                f"guess is that this lesion behaves like a **{cls_label}**-type tumor "
                f"(confidence ~{cls_conf:.2f}). This is only a computer prediction, not a "
                "confirmed diagnosis. Final diagnosis depends on full imaging and often "
                "pathology."
            )

        if user_q:
            rag_question = (
                f"{image_summary}\n\n"
                f"The user question is:\n\"{user_q}\"\n\n"
                "Using only the retrieved documents, please:\n"
                "1) Briefly restate what the image + model suggest in general, making clear "
                "that this is NOT a definitive diagnosis.\n"
                "2) Then answer **all parts** of the question in order. If the question has "
                "several sub-questions (for example: 'what type of tumor is this?', "
                "'can the patient be cured?', 'what percentage of tumor is found?'), "
                "respond to each of those parts explicitly.\n"
                "3) For numerical details like 'what percentage of tumor is found', do NOT "
                "invent numbers. If the information is not available from the documents, "
                "say that this tool cannot reliably estimate exact percentages and that "
                "radiologists and surgeons do these measurements.\n"
                "4) Keep the discussion general and safe (no patient-specific treatment "
                "prescriptions or drug doses).\n"
            )
        else:
            rag_question = (
                f"{image_summary}\n\n"
                "Using only the retrieved documents, explain in general terms what this "
                "tumor category usually is, its typical imaging features, and common "
                "management approaches. Do NOT make a patient-specific diagnosis or "
                "treatment plan."
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

    return (
        cls_label,
        f"{cls_conf:.3f}",
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
            # Left column: inputs
            with gr.Column():
                image_in = gr.Image(
                    label="MRI slice (optional)",
                    type="pil"
                )
                question_in = gr.Textbox(
                    label="Question (optional)",
                    lines=4,
                    placeholder="e.g. What does this kind of tumor usually mean?"
                )
                audience_in = gr.Radio(
                    ["Patient", "Clinician"],
                    value="Patient",
                    label="Audience"
                )
                topk_in = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=5,
                    step=1,
                    label="Top-K documents to retrieve"
                )
                run_btn = gr.Button("Run Assistant", variant="primary")

            # Middle column: classifier outputs
            with gr.Column():
                gr.Markdown("## Imaging Classification (BRISC-based)")
                cls_label_out = gr.Textbox(
                    label="Predicted tumor category (adjusted)",
                    interactive=False
                )
                cls_conf_out = gr.Textbox(
                    label="Confidence (0–1, approximate)",
                    interactive=False
                )
                overlay_out = gr.Image(
                    label="Segmentation / ROI overlay (if available)"
                )
                cls_expl_out = gr.Textbox(
                    label="Classifier explanation (rules, probabilities, notes)",
                    lines=10,
                    interactive=False
                )

            # Right column: RAG answer + traceability
            with gr.Column():
                gr.Markdown("## Document-grounded Answer")
                answer_out = gr.Textbox(
                    label="Answer",
                    lines=10,
                    interactive=False
                )
                sources_out = gr.Textbox(
                    label="Sources (guidelines, textbooks, papers)",
                    lines=6,
                    interactive=False
                )
                context_out = gr.Textbox(
                    label="Context used (debug / traceability)",
                    lines=8,
                    interactive=False
                )
                dbg_out = gr.Textbox(
                    label="RAG debug / errors",
                    lines=4,
                    interactive=False
            )

        # Wire button
        run_btn.click(
            fn=multimodal_pipeline,
            inputs=[image_in, question_in, audience_in, topk_in],
            outputs=[
                cls_label_out,
                cls_conf_out,
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
