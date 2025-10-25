# ============================================================
# 完整示例：L1 Logistic + 验证集调参与评估
# ============================================================
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_recall_curve, average_precision_score,
                             roc_auc_score, f1_score)
import matplotlib.pyplot as plt


def load_data(zero_data_path, one_data_path):
    zero_data = np.load(zero_data_path)
    one_data = np.load(one_data_path)
    zero_data = zero_data.reshape(zero_data.shape[0], -1)
    one_data = one_data.reshape(one_data.shape[0], -1)
    X = np.concatenate((zero_data, one_data), axis=0)
    y = np.concatenate((np.zeros(zero_data.shape[0]), np.ones(one_data.shape[0])), axis=0)
    return X, y


train_data_X, train_data_y = load_data(
    "/home/yangchunhao/csc/exp/p2p/cscd-ns/train_realtime/normal_token.npy",
    "/home/yangchunhao/csc/exp/p2p/cscd-ns/train_realtime/first_diff_token.npy"
)
valid_data_X, valid_data_y = load_data(
    "/home/yangchunhao/csc/exp/p2p/cscd-ns/test_realtime/normal_token.npy",
    "/home/yangchunhao/csc/exp/p2p/cscd-ns/test_realtime/first_diff_token.npy"
)

# ---------------------------
# 1) 在验证集上进行参数搜索
# ---------------------------
param_grid = [3.0, 2.0, 1.0, 0.5]
best_score = -np.inf
best_pipe = None
best_C = None
best_valid_probs = None

for C in param_grid:
    candidate_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(penalty='l1', solver='saga', max_iter=5000, C=C))
    ])
    candidate_pipe.fit(train_data_X, train_data_y)
    valid_probs = candidate_pipe.predict_proba(valid_data_X)[:, 1]
    valid_preds_05 = (valid_probs >= 0.5).astype(int)
    score = f1_score(valid_data_y, valid_preds_05)
    print(f"C={C}: validation F1 (thr=0.5) = {score:.4f}")
    if score > best_score:
        best_score = score
        best_pipe = candidate_pipe
        best_C = C
        best_valid_probs = valid_probs

print("=== Validation search done ===")
print("Best C:", best_C)
print("Best validation F1 (thr=0.5):", best_score)
print()

# ---------------------------
# 2) 基本指标（默认阈值 0.5）
# ---------------------------
valid_preds_05 = (best_valid_probs >= 0.5).astype(int)
print("=== Validation metrics at threshold 0.5 ===")
print(classification_report(valid_data_y, valid_preds_05, digits=4))
print("Confusion matrix:\n", confusion_matrix(valid_data_y, valid_preds_05))
print("ROC-AUC (valid):", roc_auc_score(valid_data_y, best_valid_probs))
print("PR-AUC (valid average precision):", average_precision_score(valid_data_y, best_valid_probs))
print()

# ---------------------------
# 3) 在验证集概率上搜索最佳阈值（使 F1 最大）
# ---------------------------
precisions, recalls, thresholds = precision_recall_curve(valid_data_y, best_valid_probs)
# thresholds length = len(precisions)-1
f1s = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-12)
best_idx = np.nanargmax(f1s)
best_thresh = thresholds[best_idx]
best_f1 = f1s[best_idx]
best_precision = precisions[:-1][best_idx]
best_recall = recalls[:-1][best_idx]

print("=== Best threshold on validation (max F1) ===")
print("best_thresh:", best_thresh)
print("best_f1:", best_f1, "precision:", best_precision, "recall:", best_recall)
print()

# 指定最佳阈值后的预测与报告
valid_preds_best = (best_valid_probs >= best_thresh).astype(int)
print("=== Validation metrics at best threshold ===")
print(classification_report(valid_data_y, valid_preds_best, digits=4))
print("Confusion matrix:\n", confusion_matrix(valid_data_y, valid_preds_best))
print("ROC-AUC (valid):", roc_auc_score(valid_data_y, best_valid_probs))
print("PR-AUC (valid average precision):", average_precision_score(valid_data_y, best_valid_probs))
print()

# ---------------------------
# 4) 可视化：PR curve 与 ROC curve（使用验证集）
# ---------------------------
# PR curve
ap = average_precision_score(valid_data_y, best_valid_probs)
precisions_valid, recalls_valid, _ = precision_recall_curve(valid_data_y, best_valid_probs)
plt.figure(figsize=(6,5))
plt.step(recalls_valid, precisions_valid, where='post')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(f'Validation Precision-Recall curve (AP={ap:.4f})')
plt.grid(True)
plt.show()
plt.savefig('PR_curve.png')

# ROC curve (画简单的 ROC 曲线)
from sklearn.metrics import roc_curve
fpr, tpr, roc_th = roc_curve(valid_data_y, best_valid_probs)
auc = roc_auc_score(valid_data_y, best_valid_probs)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f'Validation ROC curve (AUC={auc:.4f})')
plt.plot([0,1],[0,1],'--', linewidth=0.8)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('ROC_curve.png')

# ---------------------------
# 5) 查看被 L1 选中的特征（best_pipe 已在训练集上以最佳 C 拟合完成）
# ---------------------------
clf = best_pipe.named_steps['clf']
coefs = clf.coef_.ravel()
nonzero_idx = np.flatnonzero(np.abs(coefs) > 1e-5)
print("Num selected features (trained on train data):", nonzero_idx.size)
print("Selected feature indices:", nonzero_idx)

# 将选中特征映回 28x28 热图（如果你确定原特征顺序是 layer-major）
heat = np.zeros(784)
heat[nonzero_idx] = coefs[nonzero_idx]  # 或设为 1 表示被选中
heat_map = heat.reshape(28, 28)
plt.figure(figsize=(6,5))
plt.imshow(heat_map, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title('Selected features heatmap (28x28)')
plt.show()
plt.savefig('selected_features_heatmap.png')