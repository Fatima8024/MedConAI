# core_classifier.py
# Wrap BRISC classifier + segmentation into a safe function for the Gradio app.

import os
import sys
import json
import re
from typing import Any, Dict

# --- Make BrainTumorChatbot importable ---
# Adjust this path if your folder name differs
BRAINBOT_ROOT = r"C:\Fatima_Final_Bot\BrainTumorChatbot"
if BRAINBOT_ROOT not in sys.path:
    sys.path.insert(0, BRAINBOT_ROOT)

# Import your BRISC helper (this must exist in BrainTumorChatbot)
try:
    from app_brisc_segaware import classify_brisc_segaware, BRISC_CLS_PT, BRISC_SEG_PT, LABELS
except Exception as e:
    raise RuntimeError(
        f"Could not import BRISC pipeline from BrainTumorChatbot.\n"
        f"Check that app_brisc_segaware.py exists and is importable.\n"
        f"Error: {type(e).__name__}: {e}"
    )

def _to_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

def run_classifier(image) -> Dict[str, Any]:
    """
    Returns:
      adjusted_label: one of {glioma, meningioma, pituitary, no_tumor}
      confidence: top1 probability (0..1)
      status: confident | low_confidence | uncertain | error
      probs: dict(label -> prob)
      overlay: PIL image or None
      explanation: text for UI
    """
    # Always define locals so we never crash on partial failures
    raw_label = "unknown"
    adjusted_label = "unknown"
    probs_dict: Dict[str, float] = {}
    overlay = None
    dbg_text = ""
    status = "error"
    top1_label, top2_label = None, None
    top1_prob, top2_prob = 0.0, 0.0

    if image is None:
        return {
            "adjusted_label": "no_image",
            "confidence": 0.0,
            "status": "error",
            "probs": {},
            "overlay": None,
            "explanation": "No MRI image provided, classifier not run.",
        }

    # --- Run BRISC pipeline safely ---
    try:
        # Expected: raw_label, adjusted_label, probs_json, overlay_img, dbg_text
        raw_label, adjusted_label, probs_json, overlay, dbg_text = classify_brisc_segaware(image)
    except Exception as e:
        return {
            "adjusted_label": "error",
            "confidence": 0.0,
            "status": "error",
            "probs": {},
            "overlay": None,
            "explanation": f"[Classifier ERROR] {type(e).__name__}: {e}",
        }

    # --- Parse probabilities safely ---
    try:
        if isinstance(probs_json, str) and probs_json.strip():
            probs_dict = json.loads(probs_json)
        elif isinstance(probs_json, dict):
            probs_dict = probs_json
        else:
            probs_dict = {}
    except Exception:
        probs_dict = {}

    # Normalize / float-cast
    probs_dict = {k: _to_float(v, 0.0) for k, v in (probs_dict or {}).items()}

    # Sort
    sorted_probs = sorted(probs_dict.items(), key=lambda x: x[1], reverse=True)
    if sorted_probs:
        top1_label, top1_prob = sorted_probs[0]
    if len(sorted_probs) >= 2:
        top2_label, top2_prob = sorted_probs[1]

    # Strip any prefixes if your downstream code added them before
    base_adjusted = re.sub(r"^(uncertain_|low_confidence_)", "", (adjusted_label or "").strip().lower())
    base_adjusted = base_adjusted or (top1_label or "unknown")

    # --- Decide status ---
    # You can tweak these thresholds
    GAP_TH = 0.15
    LOW_TH = 0.60

    status = "confident"
    if top1_prob < LOW_TH:
        status = "low_confidence"
    if top2_label is not None and (top1_prob - top2_prob) < GAP_TH:
        # "uncertain" wins over low_confidence only if you prefer;
        # if you want low_confidence to dominate, swap ordering.
        status = "uncertain" if top1_prob >= LOW_TH else "low_confidence"

    # --- Build explanation for UI ---
    lines = []
    lines.append(f"Raw label: {raw_label}")
    lines.append(f"Adjusted label: {base_adjusted}")
    lines.append(f"Confidence (top prediction): {top1_prob:.3f}")
    if top2_label is not None:
        lines.append(f"Runner-up: {top2_label} ({top2_prob:.3f}), gap={top1_prob - top2_prob:.3f}")
    lines.append(f"Status: {status}")
    if probs_dict:
        lines.append("All Probabilities:\n" + json.dumps(probs_dict, indent=2))
    if dbg_text:
        lines.append("\nDebug / guard details:\n" + str(dbg_text))

    explanation_text = "\n".join(lines)

    return {
        "adjusted_label": base_adjusted,
        "confidence": float(top1_prob),
        "status": status,
        "probs": probs_dict,
        "overlay": overlay,
        "explanation": explanation_text,
    }
