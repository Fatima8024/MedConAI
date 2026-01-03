import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# 1) Load arrays
y_true = np.load("y_true.npy")
y_pred = np.load("y_pred.npy")

# 2) Class names in the SAME ORDER as CLASS_TO_IDX
class_names = [
    "Glioma",
    "Meningioma",
    "Pituitary lesion",
    "No tumor",
]

# 3) Print per-class precision / recall / F1 (for your thesis table)
print("Classification report:\n")
print(classification_report(y_true, y_pred, target_names=class_names, digits=3))

# 4) Confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("\nConfusion matrix:")
print(cm)
