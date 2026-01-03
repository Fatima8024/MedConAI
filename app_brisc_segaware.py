# app_brisc_segaware.py
# ---------------------------------------------------------
# BRISC2025 Tumor Classifier (Clean + Seg-aware)
# - Loads best TorchScript classifier from brisc_exports/brisc_classifier.pt
# - Optionally loads U-Net seg from brisc_exports/brisc_unet_seg.pt
# - Uses BRISC2025Loader (configs/brisc.yaml) for preprocessing
# - Adds:
#     * ROI-assisted ensemble (crop around tumor mask) with TRUST GATE
#     * Conservative no_tumor guard using seg area
#     * Uncertainty note for close top-2 probs
#
# Educational use only. Not a medical device.
# ---------------------------------------------------------

import os
import io
import json

import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn.functional as F
import gradio as gr

from utils.config import load_cfg
from data.loader_registry import get_loader  # must map "brisc2025" -> BRISC2025Loader

# -------------------------
# Paths / Config
# -------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CFG = load_cfg(os.path.join(BASE_DIR, "configs", "brisc.yaml"))

BRISC_CLS_PT = os.getenv(
    "BRISC_CLS_PT",
    os.path.join(BASE_DIR, "brisc_exports", "brisc_classifier.pt"),
)
BRISC_SEG_PT = os.getenv(
    "BRISC_SEG_PT",
    os.path.join(BASE_DIR, "brisc_exports", "brisc_unet_seg.pt"),
)

LABELS = list(CFG.inference.labels)  # ["glioma", "meningioma", "no_tumor", "pituitary"]
DEVICE = torch.device("cpu")  # keep CPU for portability; flip if you want to use cuda

# ---- Seg-aware heuristics (tunable) ----
RAW_MIN_KEEP   = 0.60   # if raw prob >= this, keep raw label
ALT_MIN        = 0.85   # alternative prob must be at least this to flip (when raw is low)
GAP_MIN        = 0.20   # alternative - raw must exceed this gap to flip

EDGE_MARGIN    = 0.10   # meningioma: bbox touches image edge within 10%
SELLA_MID_TOL  = 0.12   # pituitary: x close to midline (±12% of width)
SELLA_LOW_BAND = 0.45   # pituitary: y in lower ~45% of image


def _edge_contact(bbox, H, W, margin=EDGE_MARGIN):
    (x1, y1, x2, y2) = bbox
    return (x1 <= W * margin) or (x2 >= W * (1 - margin)) or (y1 <= H * margin) or (y2 >= H * (1 - margin))


def _near_sella(bbox, H, W, mid_tol=SELLA_MID_TOL, low_band=SELLA_LOW_BAND):
    (x1, y1, x2, y2) = bbox
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    return (abs(cx / W - 0.5) <= mid_tol) and (cy / H >= (1.0 - low_band))


def _region_supports(label, bbox, H, W):
    if label == "meningioma":
        return _edge_contact(bbox, H, W)
    if label == "pituitary":
        return _near_sella(bbox, H, W)
    return True


# -------------------------
# Loader (exact training preproc)
# -------------------------

Loader = get_loader("brisc2025")
_loader = Loader(CFG.preproc)


def preprocess_pil(img: Image.Image) -> torch.Tensor:
    """
    EXACTLY match BRISC2025Loader:
      - convert to RGB
      - gray3 or rgb based on config
      - center crop (center_crop_frac)
      - resize to input_size
      - [0,1] + per-image zscore (or mean/std) as in brisc.yaml
    """
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    arr = _loader.decode_upload(buf.getvalue())  # HxWx3
    x = _loader.to_tensor(arr)                   # 3xH'xW'
    x = _loader.normalize(x)                     # same normalization as training
    return x.unsqueeze(0).to(DEVICE)             # 1x3xH'xW'


# -------------------------
# Segmentation helpers
# -------------------------

def make_overlay(orig_pil: Image.Image, mask_224: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Overlay red transparent mask on original image."""
    if mask_224 is None:
        return orig_pil
    ow, oh = orig_pil.size
    mask_img = Image.fromarray(mask_224.astype(np.uint8) * 255, mode="L").resize(
        (ow, oh), resample=Image.NEAREST
    )
    overlay = Image.new("RGBA", (ow, oh), (0, 0, 0, 0))
    opx, mpx = overlay.load(), mask_img.load()
    for y in range(oh):
        for x in range(ow):
            if mpx[x, y] > 0:
                opx[x, y] = (255, 0, 0, int(255 * alpha))
    base = orig_pil.convert("RGBA")
    return Image.alpha_composite(base, overlay).convert("RGB")


def _largest_cc_bbox(mask: np.ndarray, pad_frac: float = 0.10):
    """mask: HxW uint8 {0,1} -> padded (x1,y1,x2,y2) or None."""
    if mask is None:
        return None
    if mask.ndim == 3:
        mask = mask.squeeze()
    m = (mask > 0).astype(np.uint8)
    if m.sum() == 0:
        return None
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num_labels <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = 1 + int(np.argmax(areas))
    x, y, w, h = (
        stats[idx, cv2.CC_STAT_LEFT],
        stats[idx, cv2.CC_STAT_TOP],
        stats[idx, cv2.CC_STAT_WIDTH],
        stats[idx, cv2.CC_STAT_HEIGHT],
    )
    H, W = m.shape
    pad = int(pad_frac * max(H, W))
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(W, x + w + pad)
    y2 = min(H, y + h + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _prep_tensor_from_np_with_loader(arr_rgb: np.ndarray) -> torch.Tensor:
    """Use same BRISC loader on an RGB numpy crop."""
    pil = Image.fromarray(arr_rgb)
    buf = io.BytesIO()
    pil.convert("RGB").save(buf, format="PNG")
    arr = _loader.decode_upload(buf.getvalue())
    x = _loader.to_tensor(arr)
    x = _loader.normalize(x)
    return x.unsqueeze(0).to(DEVICE)


# -------------------------
# Model loading
# -------------------------

_cls_model = None
_seg_model = None


def load_models():
    global _cls_model, _seg_model

    if _cls_model is None:
        if not os.path.exists(BRISC_CLS_PT):
            raise FileNotFoundError(
                f"Classifier PT not found at {BRISC_CLS_PT}.\n"
                "Run train_brisc_resnet.py / export_models.py first."
            )
        m = torch.jit.load(BRISC_CLS_PT, map_location=DEVICE)
        m.eval()
        _cls_model = m
        print("[INIT] Loaded classifier from", BRISC_CLS_PT)
        print("[INIT] Labels:", LABELS)

    if _seg_model is None and os.path.exists(BRISC_SEG_PT):
        try:
            s = torch.jit.load(BRISC_SEG_PT, map_location=DEVICE)
            s.eval()
            _seg_model = s
            print("[INIT] Loaded segmentation model from", BRISC_SEG_PT)
        except Exception as e:
            print("[WARN] Failed to load seg model:", e)
            _seg_model = None

    return _cls_model, _seg_model


# -------------------------
# Core inference
# -------------------------

def classify_brisc_segaware(img: Image.Image):
    if img is None:
        return "No image.", "No image.", "{}", None, "No image provided."

    cls_model, seg_model = load_models()
    orig = img.convert("RGB")
    x = preprocess_pil(orig)  # 1x3xHxW

    dbg_lines = []

    # --- segmentation ---
    seg_mask_224 = None
    tumor_frac = None
    if seg_model is not None:
        try:
            with torch.no_grad():
                seg_out = seg_model(x)  # [1,C,H,W] or [1,1,H,W]
            if seg_out.ndim == 4:
                if seg_out.shape[1] > 1:
                    seg_cls = torch.argmax(seg_out, dim=1)[0].cpu().numpy()
                    seg_mask_224 = (seg_cls > 0).astype(np.uint8)
                else:
                    seg_prob = torch.sigmoid(seg_out)[0, 0].cpu().numpy()
                    seg_mask_224 = (seg_prob > 0.5).astype(np.uint8)
            tumor_frac = float(seg_mask_224.mean()) if seg_mask_224 is not None else None
            if tumor_frac is not None:
                dbg_lines.append(f"Seg tumor_frac: {tumor_frac:.5f}")
        except Exception as e:
            dbg_lines.append(f"Segmentation error: {type(e).__name__}: {e}")
            seg_mask_224 = None
            tumor_frac = None
    else:
        dbg_lines.append("Seg model not loaded; running classifier only.")

    # --- classifier: whole image (raw baseline) ---
    with torch.no_grad():
        logits = cls_model(x)
        probs_base = F.softmax(logits, dim=1)[0].cpu().numpy()

    if probs_base.shape[0] != len(LABELS):
        msg = (
            f"Label mismatch: model has {probs_base.shape[0]} outputs, "
            f"but LABELS has {len(LABELS)}. Check config/export."
        )
        return msg, msg, "{}", None, msg

    raw_idx = int(np.argmax(probs_base))
    raw_label = LABELS[raw_idx]
    raw_top = float(probs_base[raw_idx])
    dbg_lines.append(f"Raw: {raw_label} ({raw_top:.3f})")

    # start with whole-image probs; may be refined by ROI
    probs = probs_base.copy()

    # -----------------------------
    # ROI-assisted ensemble (build bbox first)
    # -----------------------------
    roi_debug = None
    roi_bbox = None
    nt_idx = LABELS.index("no_tumor") if "no_tumor" in LABELS else None

    if seg_mask_224 is not None and seg_mask_224.sum() > 0:
        try:
            ow, oh = orig.size
            seg_up = Image.fromarray(seg_mask_224 * 255, mode="L").resize((ow, oh), resample=Image.NEAREST)
            seg_np = np.array(seg_up)
            bbox = _largest_cc_bbox(seg_np, pad_frac=0.10)

            if bbox is not None:
                roi_bbox = bbox
                x1, y1, x2, y2 = bbox
                full_np = np.array(orig)
                crop = full_np[y1:y2, x1:x2, :].copy()

                if crop.size > 0:
                    crop_mask = seg_np[y1:y2, x1:x2]
                    crop_masked = crop.copy()
                    crop_masked[crop_mask == 0] = (crop_masked[crop_mask == 0] * 0.25).astype(crop_masked.dtype)

                    x_roi = _prep_tensor_from_np_with_loader(crop_masked)

                    with torch.no_grad():
                        logits_roi = cls_model(x_roi)
                        probs_roi = F.softmax(logits_roi, dim=1)[0].cpu().numpy()

                    # -----------------------------
                    # ROI trust gate (prevents ROI drift)
                    # -----------------------------
                    H, W = np.array(orig).shape[:2]
                    bbox_area = max(1, (x2 - x1) * (y2 - y1))
                    bbox_frac = bbox_area / float(H * W)

                    roi_top_idx = int(np.argmax(probs_roi))
                    roi_top_lbl = LABELS[roi_top_idx]
                    roi_top_p = float(probs_roi[roi_top_idx])

                    skip_reason = None

                    # 1) bbox too tiny / too huge
                    if bbox_frac < 0.002:
                        skip_reason = f"bbox too small (bbox_frac={bbox_frac:.4f})"
                    elif bbox_frac > 0.60:
                        skip_reason = f"bbox too large (bbox_frac={bbox_frac:.4f})"

                    # 2) ROI not confident
                    elif roi_top_p < 0.70:
                        skip_reason = f"ROI not confident (top={roi_top_lbl} p={roi_top_p:.3f})"

                    # 3) Pituitary sanity: should be near sella
                    elif raw_label == "pituitary" and (not _near_sella(bbox, H, W)):
                        skip_reason = "raw pituitary but ROI bbox not near sella"

                    if skip_reason:
                        dbg_lines.append(f"Skip ROI ensemble: {skip_reason}")
                    else:
                        ROI_ALPHA = 0.25  # lower than 0.6 reduces drift
                        probs_ens = probs.copy()

                        # blend ONLY tumor classes; leave 'no_tumor' as is
                        for i in range(len(LABELS)):
                            if (nt_idx is not None) and (i == nt_idx):
                                continue
                            probs_ens[i] = (1.0 - ROI_ALPHA) * probs[i] + ROI_ALPHA * probs_roi[i]

                        # renormalize
                        s = float(probs_ens.sum())
                        if s > 0:
                            probs_ens = probs_ens / s

                        probs = probs_ens
                        roi_debug = {
                            "bbox": [int(v) for v in bbox],
                            "probs_roi": {LABELS[i]: float(probs_roi[i]) for i in range(len(LABELS))},
                        }
                        dbg_lines.append(f"ROI ensemble applied (alpha={ROI_ALPHA}) on tumor classes only.")

        except Exception as e:
            dbg_lines.append(f"ROI ensemble error: {type(e).__name__}: {e}")

    # -----------------------------
    # Decide final label from probs AFTER ROI (if any)
    # -----------------------------
    final_idx = int(np.argmax(probs))
    final_label = LABELS[final_idx]

    # -----------------------------
    # Guard: strong seg + strong raw tumor → don't let ensemble flip to no_tumor
    # -----------------------------
    if (
        tumor_frac is not None
        and tumor_frac > 0.03
        and raw_label != "no_tumor"
        and raw_top >= 0.80
        and final_label == "no_tumor"
    ):
        dbg_lines.append("Guard: strong mask + strong raw tumor → ignore ensemble no_tumor.")
        probs = probs_base.copy()
        final_idx = raw_idx
        final_label = raw_label

    # -----------------------------
    # seg-aware no_tumor guard (tiny mask only)
    # -----------------------------
    adjusted_label = final_label
    if (tumor_frac is not None) and (nt_idx is not None) and (final_label != "no_tumor"):
        NO_TUMOR_MAX_FRAC = 0.02  # <2% seg area
        CONF_CUTOFF = 0.90
        NT_RATIO_MIN = 0.6
        p_top = float(probs[final_idx])
        p_nt = float(probs[nt_idx])
        if (tumor_frac < NO_TUMOR_MAX_FRAC) and (p_top < CONF_CUTOFF) and (p_nt >= NT_RATIO_MIN * p_top):
            adjusted_label = "no_tumor"
            dbg_lines.append("Guard: tiny seg + not confident → override to no_tumor.")

    # -----------------------------
    # Conservative seg-aware flips between tumor classes (FIXED)
    # -----------------------------
    H, W = np.array(orig).shape[:2]
    order = np.argsort(probs)[::-1]
    top1_idx = int(order[0])
    top2_idx = int(order[1]) if len(order) > 1 else top1_idx

    top1_lbl = LABELS[top1_idx]
    top2_lbl = LABELS[top2_idx]
    top1_p = float(probs[top1_idx])
    top2_p = float(probs[top2_idx])

    flip_reason = "kept_raw"

    # Only consider flipping away from raw if raw confidence was low
    if raw_top < RAW_MIN_KEEP:
        # Use TOP1 as the alternative candidate (not top2)
        if (
            top1_lbl != raw_label
            and top1_p >= ALT_MIN
            and (top1_p - raw_top) >= GAP_MIN
            and roi_bbox is not None
            and _region_supports(top1_lbl, roi_bbox, H, W)
        ):
            # don't flip to meningioma unless edge-contact holds
            if not (top1_lbl == "meningioma" and not _edge_contact(roi_bbox, H, W)):
                adjusted_label = top1_lbl
                flip_reason = f"flip_to_{top1_lbl}_ALT_MIN_GAP_MIN_region_ok"
            else:
                flip_reason = "blocked_meningioma_no_edge"

    # If pituitary was raw and ROI looks sellar, never flip away
    if (raw_label == "pituitary") and (roi_bbox is not None) and _near_sella(roi_bbox, H, W):
        adjusted_label = "pituitary"
        flip_reason = "revert_pituitary_region_strong"

    # Uncertainty note (top-2 close)
    if len(order) >= 2:
        margin = float(probs[order[0]] - probs[order[1]])
        if margin < 0.12:
            dbg_lines.append(
                f"Uncertain: top-2 close ({LABELS[order[0]]} vs {LABELS[order[1]]}, Δ={margin:.3f})"
            )

    # -----------------------------
    # Build outputs
    # -----------------------------
    probs_dict = {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}
    overlay = make_overlay(orig, seg_mask_224) if seg_mask_224 is not None else None

    if roi_debug is not None:
        dbg_lines.append("ROI debug: " + json.dumps(roi_debug))

    dbg_lines.append(
        f"SegAware info: raw={raw_label}({raw_top:.3f}) → adjusted={adjusted_label} "
        f"| top1=({top1_lbl},{top1_p:.3f}) top2=({top2_lbl},{top2_p:.3f}) | reason={flip_reason}"
    )

    dbg_text = "\n".join(dbg_lines) if dbg_lines else "(no debug info)"

    return raw_label, adjusted_label, json.dumps(probs_dict, indent=2), overlay, dbg_text


# -------------------------
# Gradio UI
# -------------------------

with gr.Blocks(title="BRISC2025 Tumor Classifier (Clean + Seg-aware)") as demo:
    gr.Markdown(
        "## 🧠 BRISC2025 Tumor Classifier (Clean + Seg-aware)\n"
        "_Educational use only. Not a medical device._"
    )

    with gr.Row():
        img_in = gr.Image(type="pil", label="Upload BRISC-style MRI slice (JPG/PNG)")

    btn = gr.Button("Analyze", variant="primary")

    gr.Markdown("### Prediction (raw + adjusted)")
    out_raw = gr.Textbox(label="Predicted (raw)")
    out_adj = gr.Textbox(label="Predicted (adjusted)")

    gr.Markdown("### Class probabilities (JSON)")
    out_probs = gr.Textbox(lines=10)

    gr.Markdown("### Segmentation overlay")
    out_overlay = gr.Image()

    gr.Markdown("### Debug / Guard details")
    out_dbg = gr.Textbox(lines=10)

    btn.click(
        classify_brisc_segaware,
        inputs=img_in,
        outputs=[out_raw, out_adj, out_probs, out_overlay, out_dbg],
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, show_error=True)
