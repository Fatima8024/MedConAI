import numpy as np
from pathlib import Path
from PIL import Image

from core_classifier import run_classifier  # uses your existing classifier

# 1) Map class names to integer indices.
#    KEYS must match the SUBFOLDER NAMES inside your test folder.
CLASS_TO_IDX = {
    "glioma": 0,
    "meningioma": 1,
    "pituitary": 2,
    "no_tumor": 3,
}

# 2) Path to your test root folder (use raw string for Windows path)
TEST_ROOT = Path(r"C:\Fatima_Final_Bot\BrainTumorChatbot\brisc2025\classification_task\test")

y_true_list = []
y_pred_list = []
image_paths = []

for class_name, class_idx in CLASS_TO_IDX.items():
    class_dir = TEST_ROOT / class_name
    if not class_dir.exists():
        print(f"WARNING: folder does not exist: {class_dir}")
        continue

    # loop over all images in this class folder
    for img_path in class_dir.glob("*.*"):
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Skipping {img_path} (cannot open): {e}")
            continue

        # ground-truth label
        y_true_list.append(class_idx)
        image_paths.append(str(img_path))

        # run your classifier
        result = run_classifier(img)
        pred_label_raw = result.get("adjusted_label", "no_tumor")

        # normalise the predicted label to match keys like "glioma", "meningioma", ...
        pred_label = (
            str(pred_label_raw)
            .strip()
            .lower()
            .replace(" ", "_")
        )

        if pred_label not in CLASS_TO_IDX:
            print(f"Unknown predicted label '{pred_label_raw}' -> mapping to 'no_tumor' for {img_path}")
            pred_idx = CLASS_TO_IDX["no_tumor"]
        else:
            pred_idx = CLASS_TO_IDX[pred_label]

        y_pred_list.append(pred_idx)

# 3) Convert to numpy arrays and save
y_true = np.array(y_true_list, dtype=np.int64)
y_pred = np.array(y_pred_list, dtype=np.int64)
paths = np.array(image_paths)

np.save("y_true.npy", y_true)
np.save("y_pred.npy", y_pred)
np.save("test_image_paths.npy", paths)

print(f"Saved y_true.npy and y_pred.npy with {len(y_true)} samples.")
