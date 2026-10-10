"""Shared confusion-matrix plotting, used by both train.py (periodic snapshots
during training) and evaluate.py (final report). Forces the non-interactive
"Agg" backend before importing pyplot — required on a headless server with no
display, otherwise matplotlib can error out or hang trying to open a GUI window.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import confusion_matrix as sk_confusion_matrix  # noqa: E402


def save_confusion_matrix(labels, preds, class_names, out_path, title=None):
    cm = sk_confusion_matrix(labels, preds, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90)
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    if title:
        ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7)
    fig.colorbar(im)
    fig.tight_layout()
    out_path = str(out_path)
    import os
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return cm
