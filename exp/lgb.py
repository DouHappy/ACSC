import lightgbm as lgb
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split

def train_and_evaluate_lgbm_sklearn(X, y):
    """
    使用 LGBMClassifier (sklearn 风格) 进行二分类训练，并返回训练好的模型和验证集指标
    """
    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 定义 LGBMClassifier
    model = lgb.LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        learning_rate=0.05,
        num_leaves=31,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        n_estimators=2000,   # 类似 num_boost_round
        random_state=42,
        verbose=-1
    )

    # 训练 + 验证 (内置 early_stopping)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(stopping_rounds=50)],
    )


    # 预测验证集
    y_pred = model.predict(X_val)
    y_pred_prob = model.predict_proba(X_val)[:, 1]

    # 计算指标
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    report = classification_report(y_val, y_pred, digits=4)

    results = {
        "accuracy": acc,
        "f1_score": f1,
        "report": report
    }

    return model, results

normal_token = np.load("/home/yangchunhao/csc/exp/p2p/normal_token.npy")
normal_token = normal_token.reshape(normal_token.shape[0], -1)
error_token = np.load("/home/yangchunhao/csc/exp/p2p/first_diff_token.npy")
error_token = error_token.reshape(error_token.shape[0], -1)
X = np.concatenate((normal_token, error_token), axis=0)
y = np.concatenate((np.zeros(normal_token.shape[0]), np.ones(error_token.shape[0])), axis=0)

# 假设你已经有 X, y
model, results = train_and_evaluate_lgbm_sklearn(X, y)

print("Accuracy:", results["accuracy"])
print("F1 Score:", results["f1_score"])
print(results["report"])
