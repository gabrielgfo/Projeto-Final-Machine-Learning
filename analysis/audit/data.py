"""Carregamento unificado de dados para auditoria metodológica."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]

COLS_CURSOS = [
    "QT_ING",
    "QT_MAT",
    "QT_CONC",
    "QT_SIT_DESVINCULADO",
    "QT_SIT_TRANCADA",
    "QT_VG_TOTAL",
    "QT_VG_TOTAL_EAD",
    "QT_VG_TOTAL_NOTURNO",
    "QT_ING_FEM",
    "QT_ING_18_24",
    "QT_INSCRITO_TOTAL",
    "QT_MAT_FINANC",
    "QT_ING_FIES",
    "QT_ING_PROUNIP",
    "TP_ORGANIZACAO_ACADEMICA",
    "TP_REDE",
    "TP_CATEGORIA_ADMINISTRATIVA",
    "TP_GRAU_ACADEMICO",
    "TP_MODALIDADE_ENSINO",
    "TP_DIMENSAO",
    "IN_GRATUITO",
    "CO_CINE_AREA_GERAL",
]

CAT_FEATURES = [
    "TP_ORGANIZACAO_ACADEMICA",
    "TP_REDE",
    "TP_CATEGORIA_ADMINISTRATIVA",
    "TP_GRAU_ACADEMICO",
    "TP_MODALIDADE_ENSINO",
    "TP_DIMENSAO",
    "IN_GRATUITO",
    "CO_CINE_AREA_GERAL",
]
NUM_FEATURES_RAW = ["QT_ING", "QT_MAT", "QT_VG_TOTAL", "QT_INSCRITO_TOTAL"]
DERIVED_FEATURES = [
    "TAXA_CONCLUSAO",
    "RAZAO_ING_MAT",
    "PROPORCAO_EAD",
    "PROPORCAO_NOTURNO",
    "INDICE_FINANCIAMENTO",
    "PROPORCAO_FIES",
    "PROPORCAO_PROUNIP",
    "PROPORCAO_18_24",
    "PROPORCAO_FEM",
]
ALL_FEATURES = CAT_FEATURES + NUM_FEATURES_RAW + DERIVED_FEATURES
TARGET = "ALTO_RISCO_EVASAO"
SAMPLE_SIZE = 30_000

DATASET_CANDIDATES = [
    PROJECT_ROOT / "MICRODADOS_CADASTRO_CURSOS_2024.CSV",
    PROJECT_ROOT / "data" / "MICRODADOS_CADASTRO_CURSOS_2024.CSV",
    Path.home()
    / "Downloads"
    / "microdados_censo_da_educacao_superior_2024"
    / "dados"
    / "MICRODADOS_CADASTRO_CURSOS_2024.CSV",
]


def safe_div(a: pd.Series, b: pd.Series, fill: float = 0.0) -> np.ndarray:
    a_num = pd.to_numeric(a, errors="coerce").fillna(0)
    b_num = pd.to_numeric(b, errors="coerce").fillna(0)
    return np.where(b_num > 0, a_num / b_num, fill)


def _resolve_dataset_path() -> Path:
    for path in DATASET_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError("Dataset INEP não encontrado para auditoria.")


def load_notebook_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """
    Carrega dataset no padrão dos notebooks (taxa de evasão >= 20%).

    Nota metodológica: difere de ``data/preprocessamento.py`` (mediana em
    QT_ING>=10). Essa diferença é documentada na auditoria (seção 5.4).
    """
    path = _resolve_dataset_path()
    df = pd.read_csv(
        path,
        sep=";",
        encoding="latin1",
        low_memory=False,
        usecols=COLS_CURSOS,
    )

    count_cols = [
        "QT_ING",
        "QT_MAT",
        "QT_CONC",
        "QT_SIT_DESVINCULADO",
        "QT_SIT_TRANCADA",
        "QT_VG_TOTAL",
        "QT_VG_TOTAL_EAD",
        "QT_VG_TOTAL_NOTURNO",
        "QT_ING_FEM",
        "QT_ING_18_24",
        "QT_MAT_FINANC",
        "QT_ING_FIES",
        "QT_ING_PROUNIP",
    ]
    for col in count_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["QT_EVADIDOS"] = df["QT_SIT_DESVINCULADO"] + df["QT_SIT_TRANCADA"]
    df["TAXA_EVASAO"] = safe_div(df["QT_EVADIDOS"], df["QT_ING"]) * 100
    df[TARGET] = (df["TAXA_EVASAO"] >= 20).astype(int)

    df["TAXA_CONCLUSAO"] = safe_div(df["QT_CONC"], df["QT_MAT"]) * 100
    df["RAZAO_ING_MAT"] = safe_div(df["QT_ING"], df["QT_MAT"])
    df["PROPORCAO_EAD"] = safe_div(df["QT_VG_TOTAL_EAD"], df["QT_VG_TOTAL"])
    df["PROPORCAO_NOTURNO"] = safe_div(df["QT_VG_TOTAL_NOTURNO"], df["QT_VG_TOTAL"])
    df["INDICE_FINANCIAMENTO"] = safe_div(df["QT_MAT_FINANC"], df["QT_MAT"])
    df["PROPORCAO_FIES"] = safe_div(df["QT_ING_FIES"], df["QT_ING"])
    df["PROPORCAO_PROUNIP"] = safe_div(df["QT_ING_PROUNIP"], df["QT_ING"])
    df["PROPORCAO_18_24"] = safe_div(df["QT_ING_18_24"], df["QT_ING"])
    df["PROPORCAO_FEM"] = safe_div(df["QT_ING_FEM"], df["QT_ING"])

    df = df[df["QT_ING"] > 0].copy()
    X = df[ALL_FEATURES].copy()
    y = df[TARGET].astype(int)

    # Amostragem estratificada 50/50 (15k por classe)
    per_class = SAMPLE_SIZE // 2
    parts = []
    for cls in (0, 1):
        cls_df = X[y == cls]
        n_sample = min(per_class, len(cls_df))
        parts.append(cls_df.sample(n=n_sample, random_state=42))
    X = pd.concat(parts)
    y = y.loc[X.index]
    shuffle_idx = X.sample(frac=1, random_state=42).index
    X = X.loc[shuffle_idx].reset_index(drop=True)
    y = y.loc[shuffle_idx].reset_index(drop=True)

    return X, y


def handle_missing_mode(X: pd.DataFrame) -> pd.DataFrame:
    """Imputação por moda (missing=mode)."""
    X_filled = X.copy()
    for col in X_filled.columns:
        if X_filled[col].isnull().any():
            moda = X_filled[col].mode()
            fill_val = moda.iloc[0] if len(moda) > 0 else 0
            X_filled[col] = X_filled[col].fillna(fill_val)
    return X_filled


def fit_label_encoding(
    X_train: pd.DataFrame,
    X_other: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None, dict[str, LabelEncoder]]:
    """Ajusta label encoding nas colunas categóricas."""
    encoders: dict[str, LabelEncoder] = {}
    X_train_enc = X_train.copy()
    X_other_enc = None if X_other is None else X_other.copy()

    for col in CAT_FEATURES:
        if col not in X_train_enc.columns:
            continue
        encoder = LabelEncoder()
        train_vals = X_train_enc[col].fillna(-1).astype(str)
        encoder.fit(train_vals)
        encoders[col] = encoder
        X_train_enc[col] = encoder.transform(train_vals)
        if X_other_enc is not None:
            other_vals = X_other_enc[col].fillna(-1).astype(str)
            known = set(encoder.classes_)
            other_vals = other_vals.where(other_vals.isin(known), "-1")
            X_other_enc[col] = encoder.transform(other_vals)

    return X_train_enc, X_other_enc, encoders


def select_correlated_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    min_abs_corr: float = 0.05,
) -> list[str]:
    """Seleção por correlação absoluta com o alvo (corr)."""
    corrs = X_train.apply(
        lambda col: abs(np.corrcoef(col, y_train)[0, 1]) if col.std() > 0 else 0.0
    )
    selected = corrs[corrs >= min_abs_corr].index.tolist()
    if not selected:
        selected = [corrs.idxmax()]
    return selected


class FoldPreprocessor:
    """Pré-processamento padrão: mode | label | corr | standard."""

    def __init__(self, min_abs_corr: float = 0.05) -> None:
        self.min_abs_corr = min_abs_corr
        self.selected_features: list[str] = []
        self.scaler = StandardScaler()
        self.encoders: dict[str, LabelEncoder] = {}

    def fit_transform(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> np.ndarray:
        X_mode = handle_missing_mode(X_train)
        X_enc, _, self.encoders = fit_label_encoding(X_mode)
        self.selected_features = select_correlated_features(
            X_enc,
            y_train,
            self.min_abs_corr,
        )
        X_sel = X_enc[self.selected_features]
        return self.scaler.fit_transform(X_sel)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X_mode = handle_missing_mode(X)
        X_enc = fit_label_encoding_with_encoders(X_mode, self.encoders)
        X_sel = X_enc[self.selected_features]
        return self.scaler.transform(X_sel)


def fit_label_encoding_with_encoders(
    X: pd.DataFrame,
    encoders: dict[str, LabelEncoder],
) -> pd.DataFrame:
    """Aplica encoders já ajustados."""
    X_enc = X.copy()
    for col, encoder in encoders.items():
        if col in X_enc.columns:
            vals = X_enc[col].fillna(-1).astype(str)
            known = set(encoder.classes_)
            vals = vals.where(vals.isin(known), "-1")
            X_enc[col] = encoder.transform(vals)
    return X_enc
