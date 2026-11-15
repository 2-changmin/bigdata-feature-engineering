# STEP 01 데이터 준비 + STEP 02 EDA
import sys, json
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fe import load_data, FIG, TAB, TARGET
from plot import save, PALETTE

df = load_data()

# ---------------- STEP 01 ----------------
print("데이터 shape:", df.shape)
print(df.head())
print(df.info())

col_desc = {
    "PassengerId": ("식별자", "승객 번호"),
    "Survived": ("타겟", "생존 여부 (0: 사망, 1: 생존)"),
    "Pclass": ("범주형(순서)", "객실 등급 (1, 2, 3등석)"),
    "Name": ("문자열", "승객 이름"),
    "Sex": ("범주형", "성별"),
    "Age": ("수치형", "나이"),
    "SibSp": ("수치형", "함께 탄 형제자매/배우자 수"),
    "Parch": ("수치형", "함께 탄 부모/자녀 수"),
    "Ticket": ("문자열", "티켓 번호"),
    "Fare": ("수치형", "운임"),
    "Cabin": ("문자열", "객실 번호"),
    "Embarked": ("범주형", "승선 항구 (C, Q, S)"),
}
col_table = pd.DataFrame({
    "컬럼명": df.columns,
    "구분": [col_desc[c][0] for c in df.columns],
    "설명": [col_desc[c][1] for c in df.columns],
    "자료형": [str(df[c].dtype) for c in df.columns],
    "고유값 수": [df[c].nunique() for c in df.columns],
    "결측치 수": [df[c].isna().sum() for c in df.columns],
})
col_table.to_csv(TAB / "columns.csv", index=False, encoding="utf-8-sig")

summary = {"shape": list(df.shape)}

# ---------------- 결측치 비율 ----------------
missing = (df.isna().mean() * 100).round(2)
missing = missing[missing > 0].sort_values(ascending=False)
print("결측치 비율(%)\n", missing)
summary["missing"] = missing.to_dict()

fig, ax = plt.subplots(figsize=(6, 3))
sns.barplot(x=missing.values, y=missing.index, ax=ax, color=PALETTE[0])
for i, v in enumerate(missing.values):
    ax.text(v + 1, i, f"{v:.1f}%", va="center")
ax.set_xlim(0, 100)
ax.set_xlabel("결측치 비율 (%)")
ax.set_ylabel("")
ax.set_title("컬럼별 결측치 비율")
save(fig, FIG / "eda_missing.png")

# ---------------- 타겟 분포 ----------------
vc = df[TARGET].value_counts().sort_index()
summary["target"] = vc.to_dict()
fig, ax = plt.subplots(figsize=(4.5, 3.5))
sns.countplot(x=TARGET, data=df, ax=ax, hue=TARGET, palette=[PALETTE[3], PALETTE[0]], legend=False)
for i, v in enumerate(vc.values):
    ax.text(i, v + 8, f"{v}명 ({v / len(df) * 100:.1f}%)", ha="center")
ax.set_xticks([0, 1])
ax.set_xticklabels(["사망(0)", "생존(1)"])
ax.set_xlabel("")
ax.set_ylabel("인원 수")
ax.set_ylim(0, vc.max() * 1.15)
ax.set_title("타겟 변수(Survived) 분포")
save(fig, FIG / "eda_target.png")

# ---------------- 히스토그램 ----------------
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
for ax, col in zip(axes.ravel(), ["Age", "Fare", "SibSp", "Parch"]):
    discrete = col in ["SibSp", "Parch"]
    sns.histplot(data=df, x=col, hue=TARGET, multiple="stack", ax=ax, discrete=discrete,
                 bins=None if discrete else 30, palette=[PALETTE[3], PALETTE[0]])
    ax.set_title(f"{col} 분포 (왜도 {df[col].skew():.2f})")
    ax.set_ylabel("빈도")
    leg = ax.get_legend()
    leg.set_title("")
    for t, lab in zip(leg.get_texts(), ["사망", "생존"]):
        t.set_text(lab)
save(fig, FIG / "eda_hist.png")
summary["skew"] = df[["Age", "Fare", "SibSp", "Parch"]].skew().round(2).to_dict()

# ---------------- 박스플롯 / 이상치(IQR) ----------------
fig, axes = plt.subplots(1, 4, figsize=(12, 4))
for ax, col in zip(axes, ["Age", "Fare", "SibSp", "Parch"]):
    sns.boxplot(data=df, x=TARGET, y=col, ax=ax, hue=TARGET, palette=[PALETTE[3], PALETTE[0]], legend=False)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["사망", "생존"])
    ax.set_xlabel("")
    ax.set_title(col)
save(fig, FIG / "eda_boxplot.png")

rows = []
for col in ["Age", "Fare", "SibSp", "Parch"]:
    s = df[col].dropna()
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = ((s < low) | (s > high)).sum()
    rows.append([col, q1, q3, iqr, low, high, n_out, round(n_out / len(s) * 100, 1), s.max()])
outlier = pd.DataFrame(rows, columns=["변수", "Q1", "Q3", "IQR", "하한", "상한", "이상치 수", "이상치 비율(%)", "최댓값"])
outlier.to_csv(TAB / "outliers.csv", index=False, encoding="utf-8-sig")
print(outlier)

# ---------------- 상관관계 히트맵 ----------------
tmp = df.copy()
tmp["Sex"] = (tmp["Sex"] == "male").astype(int)       # 남자=1
corr_cols = ["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]
corr = tmp[corr_cols].corr()
corr.round(3).to_csv(TAB / "corr.csv", encoding="utf-8-sig")
summary["corr_target"] = corr["Survived"].drop("Survived").round(3).to_dict()

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True, ax=ax)
ax.set_title("변수 간 상관관계 (Sex: 남자=1)")
save(fig, FIG / "eda_heatmap.png")

# ---------------- 범주형 변수 countplot ----------------
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
rate = {}
for ax, col in zip(axes, ["Sex", "Pclass", "Embarked"]):
    sns.countplot(data=df, x=col, hue=TARGET, ax=ax, palette=[PALETTE[3], PALETTE[0]])
    r = df.groupby(col)[TARGET].mean()
    rate[col] = (r * 100).round(1).to_dict()
    ax.set_title(f"{col}별 생존 여부")
    ax.set_ylabel("인원 수")
    ax.legend(["사망", "생존"])
save(fig, FIG / "eda_countplot.png")
summary["survival_rate"] = {k: {str(a): b for a, b in v.items()} for k, v in rate.items()}

summary["age_missing_survival"] = round(df.loc[df.Age.isna(), TARGET].mean() * 100, 1)
summary["age_notna_survival"] = round(df.loc[df.Age.notna(), TARGET].mean() * 100, 1)
summary["fare_zero"] = int((df.Fare == 0).sum())
summary["duplicated"] = int(df.duplicated().sum())

json.dump(summary, open(TAB / "eda_summary.json", "w"), ensure_ascii=False, indent=2, default=int)
print(json.dumps(summary, ensure_ascii=False, indent=2, default=int))
