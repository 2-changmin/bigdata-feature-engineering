# 추가 가산점 항목: GridSearchCV, Feature Importance 시각화, SHAP, AutoML 비교
# (Pipeline 객체는 fe.py의 make_pipeline에서 모든 실험에 사용)
import sys, json, warnings
sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from fe import (load_data, split_data, make_pipeline, get_models, cv_score, test_score,
                CV, TAB, FIG, SEED, NUM_COLS, CAT_COLS)
from plot import save, PALETTE

df = load_data()
X_train, X_test, y_train, y_test = split_data(df)
models = get_models()
SETTING = dict(impute="mean", encode="onehot", scale="standard", select=False)   # Exp-1 설정
result = {}

# ---------------- 1. GridSearchCV ----------------
grids = {
    "LogisticRegression": {"model__C": [0.01, 0.1, 1, 10]},
    "RandomForest": {"model__n_estimators": [200, 500], "model__max_depth": [4, 6, 8, None],
                     "model__min_samples_leaf": [1, 3, 5], "model__n_jobs": [1]},
    "XGBoost": {"model__max_depth": [2, 3, 4], "model__learning_rate": [0.03, 0.1],
                "model__n_estimators": [100, 300], "model__subsample": [0.8, 1.0], "model__n_jobs": [1]},
    "LightGBM": {"model__num_leaves": [7, 15, 31], "model__learning_rate": [0.03, 0.1],
                 "model__n_estimators": [100, 300], "model__min_child_samples": [10, 20], "model__n_jobs": [1]},
}
rows, tuned = [], {}
for name, grid in grids.items():
    base_pipe = make_pipeline(models[name], **SETTING)
    gs = GridSearchCV(base_pipe, grid, cv=CV, scoring="accuracy", n_jobs=2)
    gs.fit(X_train, y_train)
    tuned[name] = gs.best_estimator_
    before_cv, after_cv = cv_score(base_pipe, X_train, y_train), cv_score(gs.best_estimator_, X_train, y_train)
    before_te = test_score(base_pipe, X_train, y_train, X_test, y_test)
    after_te = test_score(gs.best_estimator_, X_train, y_train, X_test, y_test)
    params = {k.replace("model__", ""): v for k, v in gs.best_params_.items() if k != "model__n_jobs"}
    rows.append({"model": name, "n_comb": len(gs.cv_results_["params"]), "best_params": json.dumps(params),
                 "cv_before": before_cv["cv_acc"], "cv_after": after_cv["cv_acc"],
                 "gap_before": before_cv["gap"], "gap_after": after_cv["gap"],
                 "test_before": before_te["accuracy"], "test_after": after_te["accuracy"],
                 "auc_before": before_te["roc_auc"], "auc_after": after_te["roc_auc"]})
    print(name, params, round(before_cv["cv_acc"], 4), "->", round(after_cv["cv_acc"], 4))
gs_df = pd.DataFrame(rows)
gs_df.to_csv(TAB / "gridsearch.csv", index=False, encoding="utf-8-sig")

# ---------------- 2. Feature Importance 시각화 (튜닝된 Random Forest) ----------------
rf_pipe = tuned["RandomForest"]
pre = rf_pipe[:-1]
rf = rf_pipe.named_steps["model"]
Z_test = pre.transform(X_test)
names = list(Z_test.columns)

impurity = pd.Series(rf.feature_importances_, index=names)
perm = permutation_importance(rf, Z_test, y_test, n_repeats=30, random_state=SEED, scoring="accuracy")
perm_mean = pd.Series(perm.importances_mean, index=names)
perm_std = pd.Series(perm.importances_std, index=names)

import shap
explainer = shap.TreeExplainer(rf)
sv = explainer.shap_values(Z_test)
sv = sv[:, :, 1] if np.ndim(sv) == 3 else (sv[1] if isinstance(sv, list) else sv)   # 생존(1) 클래스 기준
shap_mean = pd.Series(np.abs(sv).mean(axis=0), index=names)

imp_df = pd.DataFrame({"impurity": impurity, "perm": perm_mean, "perm_std": perm_std, "shap": shap_mean})
imp_df = imp_df.sort_values("impurity", ascending=False)
imp_df.to_csv(TAB / "importance_compare.csv", encoding="utf-8-sig")
result["rank_corr"] = imp_df[["impurity", "perm", "shap"]].rank(ascending=False).corr(method="spearman").round(3).to_dict()

order = imp_df.index[::-1]
fig, axes = plt.subplots(1, 3, figsize=(14, 5.5), sharey=True)
axes[0].barh(order, imp_df.loc[order, "impurity"], color=PALETTE[0])
axes[0].set_title("불순도 기반 중요도")
axes[1].barh(order, imp_df.loc[order, "perm"], xerr=imp_df.loc[order, "perm_std"], color=PALETTE[1])
axes[1].set_title("Permutation 중요도 (테스트셋)")
axes[2].barh(order, imp_df.loc[order, "shap"], color=PALETTE[2])
axes[2].set_title("SHAP 평균 절댓값")
for ax, col in zip(axes, ["impurity", "perm", "shap"]):
    err = imp_df.loc[order, "perm_std"] if col == "perm" else 0 * imp_df.loc[order, col]
    for i, (v, e) in enumerate(zip(imp_df.loc[order, col], err)):
        ax.text(max(v, 0) + e, i, f" {v:.3f}", va="center", fontsize=8)
    ax.set_xlim(min(0, imp_df[col].min() * 1.2), (imp_df[col] + (imp_df["perm_std"] if col == "perm" else 0)).max() * 1.3)
axes[1].axvline(0, color="gray", lw=0.8)
fig.suptitle("튜닝한 Random Forest의 변수 중요도 비교")
save(fig, FIG / "bonus_importance.png")

# ---------------- 3. SHAP 요약 그래프 ----------------
plt.figure()
np.random.seed(SEED)   # 점을 흩뿌리는 위치 고정
shap.summary_plot(sv, Z_test, show=False, max_display=len(names))
fig = plt.gcf()
fig.set_size_inches(8, 6)
ax = fig.axes[0]
ax.set_xlabel("SHAP 값 (생존 확률에 대한 영향)")
ax.set_title("SHAP 요약 그래프 (Random Forest, 테스트셋)")
if len(fig.axes) > 1:          # 색상 막대 라벨 한글로
    cb = fig.axes[1]
    cb.set_ylabel("변수 값")
    cb.set_yticks(cb.get_yticks())
    cb.set_yticklabels(["낮음", "높음"][:len(cb.get_yticks())])
save(fig, FIG / "bonus_shap.png")

# ---------------- 4. AutoML 비교 (FLAML) ----------------
from flaml import AutoML
from fe import SelectColumns, FillMissing, AddAgeGroup


def to_flaml(X, use_fe, ref=None):
    if use_fe:
        X = AddAgeGroup().transform(FillMissing("mean").fit(ref if ref is not None else X)
                                    .transform(SelectColumns().transform(X)))
    else:
        X = X[NUM_COLS + CAT_COLS].copy()
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype("category")
    return X


auto_rows = []
ref = SelectColumns().transform(X_train)
for label, use_fe in [("AutoML (원본 변수)", False), ("AutoML (특성 공학 변수)", True)]:
    A = to_flaml(X_train, use_fe, ref)
    B = to_flaml(X_test, use_fe, ref)
    for c in A.columns:
        if str(A[c].dtype) == "category":
            B[c] = pd.Categorical(B[c], categories=A[c].cat.categories)
    automl = AutoML()
    automl.fit(A, y_train, task="classification", metric="accuracy", time_budget=60,
               eval_method="cv", n_splits=5, seed=SEED, verbose=0, n_jobs=2)
    prob = automl.predict_proba(B)[:, 1]
    pred = (prob >= 0.5).astype(int)
    auto_rows.append({"method": label, "best_model": automl.best_estimator,
                      "accuracy": accuracy_score(y_test, pred), "precision": precision_score(y_test, pred),
                      "recall": recall_score(y_test, pred), "f1": f1_score(y_test, pred),
                      "roc_auc": roc_auc_score(y_test, prob)})
    print(label, automl.best_estimator, auto_rows[-1]["accuracy"])

for name, pipe in tuned.items():
    auto_rows.append({"method": f"직접 구성한 Pipeline + GridSearchCV", "best_model": name,
                      **test_score(pipe, X_train, y_train, X_test, y_test)})
pd.DataFrame(auto_rows).to_csv(TAB / "automl.csv", index=False, encoding="utf-8-sig")

json.dump(result, open(TAB / "bonus_info.json", "w"), ensure_ascii=False, indent=2)
print("끝")
