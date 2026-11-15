# STEP 03 특성 공학 비교 실험, STEP 04 변수 선택, STEP 05 모델 학습/평가
import sys, json, warnings
sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

from fe import (load_data, split_data, make_pipeline, make_base_pipeline, get_models,
                cv_score, test_score, TAB, FIG, SEED, TARGET, AGE_BINS, AGE_LABELS)
from plot import save, PALETTE

df = load_data()
X_train, X_test, y_train, y_test = split_data(df)
models = get_models()
print("train:", X_train.shape, "test:", X_test.shape)
info = {"n_train": len(X_train), "n_test": len(X_test),
        "train_rate": round(y_train.mean() * 100, 1), "test_rate": round(y_test.mean() * 100, 1)}

DEFAULT = dict(impute="median", encode="onehot", scale="standard", derived=True, select=False)


def run(name, settings, X=X_train, y=y_train):
    rows = []
    for label, opt in settings:
        for m_name, model in models.items():
            pipe = make_pipeline(model, **{**DEFAULT, **opt})
            r = cv_score(pipe, X, y)
            rows.append({"setting": label, "model": m_name, **r})
            print(name, label, m_name, round(r["cv_acc"], 4))
    out = pd.DataFrame(rows)
    out.to_csv(TAB / f"{name}.csv", index=False, encoding="utf-8-sig")
    return out


# ---------------- 3-1 결측치 처리 비교 ----------------
imp = run("exp_impute", [("Mean", dict(impute="mean")),
                         ("Median", dict(impute="median")),
                         ("Most Frequent", dict(impute="most_frequent"))])
# Drop NA: 결측치가 있는 행 삭제 후 같은 방식으로 평가
mask = X_train["Age"].notna() & X_train["Embarked"].notna()
info["dropna_n"] = int(mask.sum())
drop = run("exp_dropna", [("Drop NA", dict())], X=X_train[mask], y=y_train[mask])
imp = pd.concat([imp, drop], ignore_index=True)
imp.to_csv(TAB / "exp_impute.csv", index=False, encoding="utf-8-sig")

# ---------------- 3-2 인코딩 비교 ----------------
enc = run("exp_encode", [("One-Hot", dict(encode="onehot")),
                         ("Label", dict(encode="label"))])

# ---------------- 3-3 스케일링 비교 ----------------
sca = run("exp_scale", [("Standard", dict(scale="standard")),
                        ("MinMax", dict(scale="minmax")),
                        ("Robust", dict(scale="robust"))])

# ---------------- 3-4 파생 변수 ----------------
der = run("exp_derived", [("파생 변수 미사용", dict(derived=False)),
                          ("파생 변수 사용", dict(derived=True))])

# 파생 변수별 생존율 그래프
d = df.copy()
d["FamilySize"] = d["SibSp"] + d["Parch"] + 1
d["FarePerPerson"] = d["Fare"] / d["FamilySize"]
d["AgeGroup"] = pd.cut(d["Age"], bins=AGE_BINS, labels=AGE_LABELS)
d["FarePerPersonGroup"] = pd.qcut(d["FarePerPerson"], 4, labels=["하위 25%", "25~50%", "50~75%", "상위 25%"])
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
for ax, col, title in [(axes[0], "AgeGroup", "AgeGroup별 생존율 (나이 결측 제외)"),
                       (axes[1], "FamilySize", "FamilySize별 생존율"),
                       (axes[2], "FarePerPersonGroup", "FarePerPerson 구간별 생존율")]:
    r = d.groupby(col, observed=True)[TARGET].agg(["mean", "size"])
    sns.barplot(x=r.index.astype(str), y=r["mean"] * 100, ax=ax, color=PALETTE[0])
    for i, (m, n) in enumerate(zip(r["mean"], r["size"])):
        ax.text(i, m * 100 + 1.5, f"{m*100:.0f}%\n(n={n})", ha="center", fontsize=8)
    ax.set_ylim(0, 110)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("생존율 (%)")
save(fig, FIG / "fe_derived.png")
info["derived_rate"] = {
    "AgeGroup": d.groupby("AgeGroup", observed=True)[TARGET].mean().mul(100).round(1).to_dict(),
    "FamilySize": d.groupby("FamilySize")[TARGET].mean().mul(100).round(1).to_dict(),
    "FarePerPersonGroup": d.groupby("FarePerPersonGroup", observed=True)[TARGET].mean().mul(100).round(1).to_dict(),
}

# ---------------- STEP 04 변수 선택 (Random Forest 중요도) ----------------
importance = {}
for enc_name in ["onehot", "label"]:
    pre = make_pipeline(models["LogisticRegression"], encode=enc_name)[:-1]   # 모델 앞까지만
    Z = pre.fit_transform(X_train, y_train)
    rf = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1).fit(Z, y_train)
    imp_s = pd.Series(rf.feature_importances_, index=Z.columns).sort_values(ascending=False)
    imp_s.to_csv(TAB / f"fs_importance_{enc_name}.csv", header=["importance"], encoding="utf-8-sig")
    importance[enc_name] = imp_s

# 상위 N개를 몇 개로 할지: 후보 N마다 4개 모델의 CV 정확도 평균을 비교
n_rows = []
for enc_name, cands in [("onehot", [4, 6, 8, 10, 12]), ("label", [3, 5, 7])]:
    for n in cands:
        for m_name, model in models.items():
            pipe = make_pipeline(model, **{**DEFAULT, "encode": enc_name, "select": True, "n_select": n})
            r = cv_score(pipe, X_train, y_train)
            n_rows.append({"encode": enc_name, "N": n, "model": m_name, **r})
            print("N", enc_name, n, m_name, round(r["cv_acc"], 4))
n_df = pd.DataFrame(n_rows)
n_df.to_csv(TAB / "fs_n_search.csv", index=False, encoding="utf-8-sig")
n_mean = n_df.groupby(["encode", "N"])["cv_acc"].mean()
N_ONEHOT = int(n_mean["onehot"].idxmax())
N_LABEL = int(n_mean["label"].idxmax())
info["N_onehot"], info["N_label"] = N_ONEHOT, N_LABEL
info["n_feat_onehot"], info["n_feat_label"] = len(importance["onehot"]), len(importance["label"])
info["selected_onehot"] = list(importance["onehot"].index[:N_ONEHOT])
info["selected_label"] = list(importance["label"].index[:N_LABEL])
print("선택된 N:", N_ONEHOT, N_LABEL)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, enc_name, n, title in [(axes[0], "onehot", N_ONEHOT, "One-Hot 인코딩 기준"),
                               (axes[1], "label", N_LABEL, "Label 인코딩 기준")]:
    s = importance[enc_name]
    colors = [PALETTE[0] if i < n else "lightgray" for i in range(len(s))]
    sns.barplot(x=s.values, y=s.index, ax=ax, palette=colors, hue=s.index, legend=False)
    ax.set_title(f"{title} (선택: 상위 {n}개)")
    ax.set_xlabel("중요도")
    ax.set_ylabel("")
save(fig, FIG / "fs_importance.png")

# ---------------- 실험 비교표 (Base, Exp-1, Exp-2, Exp-3) ----------------
EXP = {
    "Exp-1": dict(impute="mean", encode="onehot", scale="standard", select=False),
    "Exp-2": dict(impute="median", encode="label", scale="minmax", select=True, n_select=N_LABEL),
    "Exp-3": dict(impute="most_frequent", encode="onehot", scale="robust", select=True, n_select=N_ONEHOT),
}
rows = []
for m_name, model in models.items():
    pipe = make_base_pipeline(model, m_name)
    rows.append({"exp": "Base", "model": m_name, "n_features": 5, **cv_score(pipe, X_train, y_train),
                 **test_score(pipe, X_train, y_train, X_test, y_test)})
    print("Base", m_name)
for e, opt in EXP.items():
    n_feat = opt["n_select"] if opt["select"] else info["n_feat_" + opt["encode"]]
    for m_name, model in models.items():
        pipe = make_pipeline(model, **{**DEFAULT, **opt})
        rows.append({"exp": e, "model": m_name, "n_features": n_feat, **cv_score(pipe, X_train, y_train),
                     **test_score(pipe, X_train, y_train, X_test, y_test)})
        print(e, m_name)
main = pd.DataFrame(rows)
main.to_csv(TAB / "main_result.csv", index=False, encoding="utf-8-sig")

# ---------------- 변수 선택 전/후 비교 (Exp-2, Exp-3 설정에서 선택만 끄고 켬) ----------------
rows = []
for e in ["Exp-2", "Exp-3"]:
    for sel in [False, True]:
        opt = {**EXP[e], "select": sel}
        for m_name, model in models.items():
            pipe = make_pipeline(model, **{**DEFAULT, **opt})
            rows.append({"exp": e, "select": "후" if sel else "전", "model": m_name,
                         **cv_score(pipe, X_train, y_train),
                         **test_score(pipe, X_train, y_train, X_test, y_test)})
fs = pd.DataFrame(rows)
fs.to_csv(TAB / "fs_before_after.csv", index=False, encoding="utf-8-sig")

json.dump(info, open(TAB / "info.json", "w"), ensure_ascii=False, indent=2, default=str)
print("끝")
