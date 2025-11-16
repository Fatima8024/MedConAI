# core_classifier.py
# Wrapper around BRISC2025 classifier from BrainTumorChatbot/app_brisc_segaware.py

import json, sys
from importlib.machinery import SourceFileLoader
from pathlib import Path
from typing import Any, Dict, Optional

from PIL import Image


# --- Load app_brisc_segaware directly from BrainTumorChatbot -----------------

PROJECT_DIR = Path(__file__).resolve().parent        # C:\Fatima_Final_Bot\MultimodalAssistant
ROOT_DIR = PROJECT_DIR.parent                   # C:\Fatima_Final_Bot
BRAIN_CHATBOT_DIR = ROOT_DIR / "BrainTumorChatbot"   # C:\Fatima_Final_Bot\BrainTumorChatbot

if not BRAIN_CHATBOT_DIR.is_dir():
    raise FileNotFoundError(
        f"Expected BrainTumorChatbot folder at {BRAIN_CHATBOT_DIR}, but it was not found."
    )

# Make sure imports like 'import utils' work inside that project
if str(BRAIN_CHATBOT_DIR) not in sys.path:
    sys.path.insert(0, str(BRAIN_CHATBOT_DIR))

APP_BRISC_PATH = BRAIN_CHATBOT_DIR / "app_brisc_segaware.py"

if not APP_BRISC_PATH.is_file():
    raise FileNotFoundError(
        f"Expected app_brisc_segaware.py at {APP_BRISC_PATH}, but it was not found."
    )

# Load the module from its file path with a unique name
_bt_cls_module = SourceFileLoader(
    "bt_app_brisc_segaware", str(APP_BRISC_PATH)
).load_module()

# This should exist in that file
classify_brisc_segaware = _bt_cls_module.classify_brisc_segaware


# --- Public API -----------------------------------------------------------------

def run_classifier(image: Optional[Image.Image]) -> Dict[str, Any]:
    """
    Run the BRISC seg-aware classifier and return a structured result.

    Parameters
    ----------
    image : PIL.Image.Image or None
        MRI slice as a PIL image. If None, returns a "no image" result.

    Returns
    -------
    dict with keys:
        - raw_label: str
        - adjusted_label: str
        - confidence: float (probability of adjusted_label, if available)
        - probs: dict[label -> float]
        - overlay: PIL.Image.Image or None
        - explanation: str (human-readable debug/explain text)
    """
    if image is None:
        return {
            "raw_label": "no_image",
            "adjusted_label": "no_image",
            "confidence": 0.0,
            "probs": {},
            "overlay": None,
            "explanation": "No image provided.",
        }

    # From BrainTumorChatbot/app_brisc_segaware.py:
    #   raw_label, adjusted_label, probs_json, overlay_img, dbg_text
    raw_label, adjusted_label, probs_json, overlay_img, dbg_text = classify_brisc_segaware(
        image
    )

    # Parse probabilities JSON
    probs_dict = {}
    try:
        if isinstance(probs_json, str):
            probs_dict = json.loads(probs_json)
        elif isinstance(probs_json, dict):
            probs_dict = probs_json
    except Exception:
        probs_dict = {}

    # Confidence = probability of adjusted_label if available
    confidence = 0.0
    if isinstance(probs_dict, dict) and adjusted_label in probs_dict:
        try:
            confidence = float(probs_dict[adjusted_label])
        except Exception:
            confidence = 0.0

    expl_lines = [
        f"Raw label: {raw_label}",
        f"Adjusted label: {adjusted_label}",
        f"Confidence (adjusted label): {confidence:.3f}",
        "Probabilities:",
        json.dumps(probs_dict, indent=2),
        "",
        "Debug / guard details:",
        dbg_text or "(no debug info)",
    ]
    explanation_text = "\n".join(expl_lines)

    return {
        "raw_label": raw_label,
        "adjusted_label": adjusted_label,
        "confidence": confidence,
        "probs": probs_dict,
        "overlay": overlay_img,
        "explanation": explanation_text,
    }
