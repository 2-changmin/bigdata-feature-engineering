# 전처리 / 파이프라인 / 평가에 공통으로 쓰는 함수 모음
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "titanic.csv"
FIG = ROOT / "outputs" / "figures"
TAB = ROOT / "outputs" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

TARGET = "Survived"
NUM_COLS = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
CAT_COLS = ["Sex", "Embarked"]


def load_data():
    return pd.read_csv(DATA)


def split_data(df):
    # 8:2 층화 분할. 테스트셋은 마지막 성능 비교표에서만 사용
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)


# ---------------------------------------------------------------
# 사용자 정의 변환기
# ---------------------------------------------------------------
class SelectColumns(BaseEstimator, TransformerMixin):
    """사용할 컬럼만 남기고 FamilySize, FarePerPerson 파생 변수를 추가"""

    def __init__(self, derived=True):
        self.derived = derived

    def fit(self, X, y=None):
        return self

    def __sklearn_is_fitted__(self):
        return True

    def transform(self, X):
        out = X[NUM_COLS + CAT_COLS].copy()
        if self.derived:
            out["FamilySize"] = X["SibSp"] + X["Parch"] + 1
            out["FarePerPerson"] = X["Fare"] / out["FamilySize"]
        return out


class FillMissing(BaseEstimator, TransformerMixin):
    """결측치 대체. 수치형은 strategy(mean/median/most_frequent)를 따르고,
    범주형(Embarked)은 평균/중앙값을 구할 수 없으므로 항상 최빈값으로 채운다."""

    def __init__(self, strategy="median"):
        self.strategy = strategy

    def fit(self, X, y=None):
        self.values_ = {}
        for c in X.columns:
            if pd.api.types.is_numeric_dtype(X[c]):
                if self.strategy == "mean":
                    self.values_[c] = X[c].mean()
                elif self.strategy == "median":
                    self.values_[c] = X[c].median()
                else:
                    self.values_[c] = X[c].mode()[0]
            else:
                self.values_[c] = X[c].mode()[0]
        return self

    def transform(self, X):
        return X.fillna(self.values_)


AGE_BINS = [-1, 12, 19, 34, 59, 200]
AGE_LABELS = ["Child", "Teen", "Young", "Middle", "Senior"]


class AddAgeGroup(BaseEstimator, TransformerMixin):
    """결측치를 채운 나이로 연령대(AgeGroup) 생성 (범주형 변수)"""

    def __init__(self, use=True):
        self.use = use

    def fit(self, X, y=None):
        return self

    def __sklearn_is_fitted__(self):
        return True

    def transform(self, X):
        X = X.copy()
        if self.use:
            X["AgeGroup"] = pd.cut(X["Age"], bins=AGE_BINS, labels=AGE_LABELS).astype(str)
        return X


# ---------------------------------------------------------------
# 파이프라인 생성
# ---------------------------------------------------------------
def get_encoder(name):
    if name == "onehot":
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    # Label Encoding: 입력 변수에는 LabelEncoder 대신 OrdinalEncoder를 사용
    return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)


def get_scaler(name):
    return {"standard": StandardScaler(), "minmax": MinMaxScaler(), "robust": RobustScaler()}[name]


def make_pipeline(model, impute="median", encode="onehot", scale="standard",
                  derived=True, select=False, n_select=8):
    num = NUM_COLS + (["FamilySize", "FarePerPerson"] if derived else [])
    cat = CAT_COLS + (["AgeGroup"] if derived else [])

    ct = ColumnTransformer([
        ("num", get_scaler(scale), num),
        ("cat", get_encoder(encode), cat),
    ], verbose_feature_names_out=False)
    ct.set_output(transform="pandas")

    steps = [
        ("columns", SelectColumns(derived=derived)),
        ("impute", FillMissing(strategy=impute)),
        ("age_group", AddAgeGroup(use=derived)),
        ("encode_scale", ct),
    ]
    if select:
        # Random Forest 중요도 상위 n_select개만 사용
        rf = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)
        steps.append(("select", SelectFromModel(rf, max_features=n_select, threshold=-np.inf)))
    steps.append(("model", clone(model)))
    return Pipeline(steps)


def make_base_pipeline(model, model_name):
    # Base: 결측치 처리/인코딩/스케일링/변수선택 없음 -> 원본 수치형 변수만 사용
    steps = [("columns", RawNumeric())]
    if model_name == "LogisticRegression":
        # 로지스틱 회귀는 NaN이 있으면 학습이 안 돼서 0으로만 채움
        steps.append(("fill0", SimpleImputer(strategy="constant", fill_value=0)))
    steps.append(("model", clone(model)))
    return Pipeline(steps)


class RawNumeric(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def __sklearn_is_fitted__(self):
        return True

    def transform(self, X):
        return X[NUM_COLS]


# ---------------------------------------------------------------
# 모델
# ---------------------------------------------------------------
def get_models():
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=SEED),
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
                                 random_state=SEED, n_jobs=2, eval_metric="logloss"),
        "LightGBM": LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15,
                                   random_state=SEED, n_jobs=2, verbose=-1),
    }


# ---------------------------------------------------------------
# 평가
# ---------------------------------------------------------------
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def cv_score(pipe, X, y):
    """학습 데이터 5-Fold 교차검증 (train 점수도 같이 받아서 과적합 정도 확인)"""
    r = cross_validate(pipe, X, y, cv=CV, scoring=["accuracy", "f1", "roc_auc"],
                       return_train_score=True)
    return {
        "cv_acc": r["test_accuracy"].mean(),
        "cv_acc_std": r["test_accuracy"].std(),
        "cv_f1": r["test_f1"].mean(),
        "cv_auc": r["test_roc_auc"].mean(),
        "train_acc": r["train_accuracy"].mean(),
        "gap": r["train_accuracy"].mean() - r["test_accuracy"].mean(),
    }


def test_score(pipe, X_tr, y_tr, X_te, y_te):
    p = clone(pipe).fit(X_tr, y_tr)
    pred = p.predict(X_te)
    prob = p.predict_proba(X_te)[:, 1]
    return {
        "accuracy": accuracy_score(y_te, pred),
        "precision": precision_score(y_te, pred),
        "recall": recall_score(y_te, pred),
        "f1": f1_score(y_te, pred),
        "roc_auc": roc_auc_score(y_te, prob),
    }
