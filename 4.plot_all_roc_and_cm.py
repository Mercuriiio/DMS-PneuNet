"""
图1: 所有方法 ROC 曲线对比 -> figures/all_methods_roc.png
图2: Ours 混淆矩阵 (3x3)  -> figures/ours_confusion_matrix.png
300 DPI, PNG, 无标题

3类分类 (Normal=0, CP=1, NCP=2)
ROC: One-vs-Rest macro-average
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import label_binarize

_DIR     = os.path.dirname(os.path.abspath(__file__))
PRED_DIR = os.path.join(_DIR, "fold_predictions")
FIG_DIR  = os.path.join(_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

N_CLASSES = 3
CLASSES   = ["Normal", "CP", "NCP"]

# ── 方法配置：全部方法，Ours 最后绘制覆盖在最上层 ──────────────────────────
METHODS = [
    ("AutoCOPD",        "AutoCOPD",           "#A9A9A9"),
    ("DenseNetWSO",     "DenseNetWSO",         "#C0C0C0"),
    ("CNN_LSTM",        "CNN-LSTM",            "#808080"),
    ("Conv3D",          "Conv3D",              "#696969"),
    ("ResNet_LSTM",     "ResNet-LSTM",         "#B0B0B0"),
    ("SwinTrans_LSTM",  "SwinTrans-LSTM",      "#787878"),
    ("ViT_LSTM",        "ViT-LSTM",            "#909090"),
    ("ORACLE_CT",       "ORACLE-CT",           "#5B9BD5"),
    ("Lung_Nodule_SSM", "Lung-Nodule-SSM",     "#ED7D31"),
    ("Ours",            "DMS-PneuNet (Ours)",  "#FF0000"),
]

# ── 加载可用的折叠预测（只用 fold1 也可跑）──────────────────────────────────
def load_all_folds(folder):
    """
    尝试加载 fold1~fold5，有几折用几折。
    y_score shape: (N, 3) — 三类 softmax 概率
    """
    path = os.path.join(PRED_DIR, folder)
    if not os.path.isdir(path):
        return None, None

    y_true_list, y_score_list = [], []
    for fold in range(1, 6):
        fp_true  = os.path.join(path, f"fold{fold}_y_true.npy")
        fp_score = os.path.join(path, f"fold{fold}_y_score.npy")
        if os.path.exists(fp_true) and os.path.exists(fp_score):
            yt = np.load(fp_true)
            ys = np.load(fp_score)
            # 兼容旧版二分类 score (N,) 或 (N,2)
            if ys.ndim == 1:
                ys = np.column_stack([1 - ys, ys, np.zeros_like(ys)])
            elif ys.shape[1] == 2:
                ys = np.column_stack([1 - ys[:, 1], ys[:, 1], np.zeros(len(ys))])
            y_true_list.append(yt)
            y_score_list.append(ys)

    if not y_true_list:
        return None, None
    return np.concatenate(y_true_list), np.concatenate(y_score_list)

# ── 计算 OVR macro-average ROC ──────────────────────────────────────────────
def compute_macro_roc(y_true, y_score):
    """
    One-vs-Rest macro-average ROC。
    返回 mean_fpr, mean_tpr, macro_auc
    """
    y_bin = label_binarize(y_true, classes=list(range(N_CLASSES)))  # (N, 3)
    mean_fpr = np.linspace(0, 1, 300)
    tprs, aucs = [], []
    for c in range(N_CLASSES):
        fpr, tpr, _ = roc_curve(y_bin[:, c], y_score[:, c])
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        aucs.append(auc(fpr, tpr))
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    macro_auc = np.mean(aucs)
    return mean_fpr, mean_tpr, macro_auc

# ══════════════════════════════════════════════════════════════════════════════
# 图1: 所有方法 ROC 曲线
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 6))
plotted = 0

for folder, label, color in METHODS:
    y_true, y_score = load_all_folds(folder)
    if y_true is None:
        print(f"  [SKIP] {folder}: 无预测文件，跳过")
        continue

    fpr, tpr, roc_auc = compute_macro_roc(y_true, y_score)

    lw     = 2.5 if folder == "Ours" else 1.4
    zorder = 10  if folder == "Ours" else 2
    alpha  = 1.0 if folder == "Ours" else 0.85

    ax.plot(fpr, tpr, color=color, linewidth=lw, zorder=zorder, alpha=alpha,
            label=f"{label} (AUC = {roc_auc:.3f})")
    plotted += 1
    print(f"  [ROC] {label:<25}  AUC = {roc_auc:.4f}")

ax.plot([0, 1], [0, 1], linestyle="--", color="#AAAAAA",
        linewidth=1.2, zorder=1, label="Random Classifier")

ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.02])
ax.set_xlabel("False Positive Rate", fontsize=20)
ax.set_ylabel("True Positive Rate",  fontsize=20)
ax.legend(loc="lower right", fontsize=12, framealpha=0.9)
ax.grid(True, linestyle="--", alpha=0.35)
plt.tight_layout()

out1 = os.path.join(FIG_DIR, "all_methods_roc.png")
plt.savefig(out1, dpi=300, bbox_inches="tight")
print(f"\nSaved: {out1}  ({plotted} 个方法)")
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# 图2: Ours 混淆矩阵 (3x3)
# ══════════════════════════════════════════════════════════════════════════════
y_true, y_score = load_all_folds("Ours")
if y_true is None:
    print("[WARN] 未找到 Ours 预测文件，跳过混淆矩阵")
else:
    y_pred = np.argmax(y_score, axis=1)   # 取概率最大的类
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")

    tick_marks = np.arange(N_CLASSES)
    ax.set_xticks(tick_marks); ax.set_xticklabels(CLASSES, fontsize=12)
    ax.set_yticks(tick_marks); ax.set_yticklabels(CLASSES, fontsize=12,
                                                   rotation=90, va="center")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            pct = cm[i, j] / cm[i].sum() * 100
            ax.text(j, i, f"{cm[i, j]}\n({pct:.1f}%)",
                    ha="center", va="center", fontsize=13,
                    color="white" if cm[i, j] > thresh else "black")

    ax.set_xlabel("Predicted Label", fontsize=12, labelpad=8)
    ax.set_ylabel("True Label",      fontsize=12, labelpad=8)
    ax.tick_params(axis="both", length=0)
    plt.tight_layout(pad=1.2)

    out2 = os.path.join(FIG_DIR, "ours_confusion_matrix.png")
    plt.savefig(out2, dpi=300, bbox_inches="tight")
    print(f"Saved: {out2}")
    plt.close()

print("\nDone.")
