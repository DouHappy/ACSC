# ============================================================
# 完整示例：GridSearch L1 Logistic + 输出评估指标（OOF 基准）
# ============================================================
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict, cross_validate
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_recall_curve, average_precision_score,
                             roc_auc_score, f1_score, precision_score, recall_score)
import matplotlib.pyplot as plt

normal_token = np.load("/home/yangchunhao/csc/exp/p2p/normal_token.npy")
normal_token = normal_token.reshape(normal_token.shape[0], -1)
error_token = np.load("/home/yangchunhao/csc/exp/p2p/first_diff_token.npy")
error_token = error_token.reshape(error_token.shape[0], -1)
X = np.concatenate((normal_token, error_token), axis=0)
y = np.concatenate((np.zeros(normal_token.shape[0]), np.ones(error_token.shape[0])), axis=0)

# ---------------------------
# 1) Pipeline 与 GridSearch
# ---------------------------
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(penalty='l1', solver='saga', max_iter=5000))
])

param_grid = {'clf__C': [3.0, 2.0, 1.0, 0.5]}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

gs = GridSearchCV(pipe, param_grid, scoring='f1', cv=cv, n_jobs=-1, return_train_score=False)
gs.fit(X, y)

print("=== GridSearch done ===")
print("Best params:", gs.best_params_)
print("Best CV F1 (mean across folds):", gs.best_score_)

# ---------------------------
# 2) 使用 Best Estimator 做 OOF 预测（更可靠的 CV 层面评估）
#    cross_val_predict 会在每一折上 fit 再 predict（不会数据泄露）
# ---------------------------
best_pipe = gs.best_estimator_

# 得到对每个样本的 oof 概率 (out-of-fold probabilities)
oof_probs = cross_val_predict(best_pipe, X, y, cv=cv, method='predict_proba', n_jobs=-1)[:, 1]
# 也可以得到 oof 直接预测（默认阈值 0.5）
oof_preds_05 = (oof_probs >= 0.5).astype(int)

# ---------------------------
# 3) 基本指标（默认阈值 0.5）
# ---------------------------
print("=== Metrics at threshold 0.5 ===")
print(classification_report(y, oof_preds_05, digits=4))
print("Confusion matrix:\n", confusion_matrix(y, oof_preds_05))
print("ROC-AUC (oof):", roc_auc_score(y, oof_probs))
print("PR-AUC (average precision):", average_precision_score(y, oof_probs))
print()

# ---------------------------
# 4) 在 OOF 概率上搜索最佳阈值（使 F1 最大）
# ---------------------------
precisions, recalls, thresholds = precision_recall_curve(y, oof_probs)
# thresholds length = len(precisions)-1
f1s = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-12)
best_idx = np.nanargmax(f1s)
best_thresh = thresholds[best_idx]
best_f1 = f1s[best_idx]
best_precision = precisions[:-1][best_idx]
best_recall = recalls[:-1][best_idx]

print("=== Best threshold on OOF (max F1) ===")
print("best_thresh:", best_thresh)
print("best_f1:", best_f1, "precision:", best_precision, "recall:", best_recall)
print()

# 指定最佳阈值后的预测与报告
oof_preds_best = (oof_probs >= best_thresh).astype(int)
print("=== Metrics at best threshold ===")
print(classification_report(y, oof_preds_best, digits=4))
print("Confusion matrix:\n", confusion_matrix(y, oof_preds_best))
print("ROC-AUC (oof):", roc_auc_score(y, oof_probs))
print("PR-AUC (average precision):", average_precision_score(y, oof_probs))
print()

# ---------------------------
# 5) 可视化：PR curve 与 ROC curve
# ---------------------------
# PR curve
ap = average_precision_score(y, oof_probs)
plt.figure(figsize=(6,5))
plt.step(recalls, precisions, where='post')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(f'Precision-Recall curve (AP={ap:.4f})')
plt.grid(True)
plt.show()
plt.savefig('PR_curve.png')

# ROC curve (画简单的 ROC 曲线)
from sklearn.metrics import roc_curve
fpr, tpr, roc_th = roc_curve(y, oof_probs)
auc = roc_auc_score(y, oof_probs)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f'ROC curve (AUC={auc:.4f})')
plt.plot([0,1],[0,1],'--', linewidth=0.8)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('ROC_curve.png')

# ---------------------------
# 6) 如果你关心每一折的结果，可以用 cross_validate 输出各项得分的均值与 std
# ---------------------------
scoring = ['precision', 'recall', 'f1', 'roc_auc', 'average_precision']
cv_res = cross_validate(best_pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=False)
for k in scoring:
    scores = cv_res[f'test_{k}']
    print(f"{k}: mean={scores.mean():.4f}, std={scores.std():.4f}")
print()

# ---------------------------
# 7) 查看被 L1 选中的特征（在整个数据上训练的 best_pipe）
#    注意：best_pipe 已经在 GridSearch 完成后被 refit 到全部数据（默认 refit=True）
# ---------------------------
clf = best_pipe.named_steps['clf']
coefs = clf.coef_.ravel()
nonzero_idx = np.flatnonzero(np.abs(coefs) > 1e-5)
print("Num selected features (trained on full data):", nonzero_idx.size)
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