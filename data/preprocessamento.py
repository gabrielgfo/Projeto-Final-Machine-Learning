"""Carregamento e preparação inicial dos microdados do Censo da Educação Superior."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_CANDIDATES = [
    PROJECT_ROOT / "MICRODADOS_CADASTRO_CURSOS_2024.CSV",
    PROJECT_ROOT / "data" / "MICRODADOS_CADASTRO_CURSOS_2024.CSV",
    Path.home()
    / "Downloads"
    / "microdados_censo_da_educacao_superior_2024"
    / "dados"
    / "MICRODADOS_CADASTRO_CURSOS_2024.CSV",
]

COLUNAS_IDENTIFICADORAS = {
    "NU_ANO_CENSO",
    "NO_REGIAO",
    "NO_UF",
    "SG_UF",
    "NO_MUNICIPIO",
    "NO_CURSO",
    "CO_CURSO",
    "CO_IES",
    "NO_CINE_ROTULO",
    "CO_CINE_ROTULO",
    "NO_CINE_AREA_GERAL",
    "NO_CINE_AREA_ESPECIFICA",
    "NO_CINE_AREA_DETALHADA",
}

COLUNAS_VAZAMENTO = {
    "QT_SIT_DESVINCULADO",
    "QT_SIT_TRANCADA",
    "QT_SIT_TRANSFERIDO",
    "QT_SIT_FALECIDO",
    "taxa_evasao",
}

PREFIXOS_VAZAMENTO = ("QT_MAT", "QT_CONC")


def _resolver_caminho_dataset() -> Path:
    """Localiza o arquivo de microdados do cadastro de cursos."""
    for caminho in DATASET_CANDIDATES:
        if caminho.exists():
            logger.info("Dataset encontrado em: %s", caminho)
            return caminho

    caminhos = "\n".join(f"  - {c}" for c in DATASET_CANDIDATES)
    raise FileNotFoundError(
        "Arquivo MICRODADOS_CADASTRO_CURSOS_2024.CSV não encontrado. "
        f"Caminhos verificados:\n{caminhos}"
    )


def _carregar_dados_brutos() -> pd.DataFrame:
    """Carrega o CSV de microdados com tratamento de erros."""
    caminho = _resolver_caminho_dataset()
    try:
        df = pd.read_csv(
            caminho,
            sep=";",
            encoding="latin-1",
            low_memory=False,
        )
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise RuntimeError(f"Falha ao carregar o dataset em {caminho}") from exc

    logger.info("Dataset carregado com %d registros e %d colunas.", len(df), len(df.columns))
    return df


def _calcular_alvo(df: pd.DataFrame) -> pd.DataFrame:
    """Define a variável alvo binária de alto risco de evasão por curso."""
    if "QT_MAT" not in df.columns or "QT_SIT_DESVINCULADO" not in df.columns:
        raise ValueError(
            "Colunas QT_MAT e QT_SIT_DESVINCULADO são necessárias para o alvo."
        )

    denominador = df["QT_MAT"] + df["QT_SIT_DESVINCULADO"]
    df = df.copy()
    df["taxa_evasao"] = np.where(
        denominador > 0,
        df["QT_SIT_DESVINCULADO"] / denominador,
        np.nan,
    )

    df = df[df["QT_ING"] >= 10]
    df = df[df["QT_MAT"] > 0]
    df = df[df["taxa_evasao"].notna()]

    limiar = df["taxa_evasao"].median()
    df["alto_risco_evasao"] = (df["taxa_evasao"] >= limiar).astype(int)

    logger.info(
        "Alvo definido com limiar de taxa de evasão %.4f. "
        "Distribuição: %s",
        limiar,
        df["alto_risco_evasao"].value_counts().to_dict(),
    )
    return df


def _selecionar_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa atributos preditivos numéricos e variável alvo."""
    colunas_excluidas = set(COLUNAS_IDENTIFICADORAS) | set(COLUNAS_VAZAMENTO)
    colunas_excluidas.update(
        coluna
        for coluna in df.columns
        if coluna.startswith(PREFIXOS_VAZAMENTO)
    )
    colunas_excluidas.add("alto_risco_evasao")

    features = df.drop(columns=[c for c in colunas_excluidas if c in df.columns])
    features = features.select_dtypes(include=[np.number])
    features = features.replace([np.inf, -np.inf], np.nan).fillna(0)

    alvo = df["alto_risco_evasao"].astype(int)
    features = features.loc[alvo.index]

    if features.empty:
        raise ValueError("Nenhuma feature numérica disponível após a seleção.")

    logger.info("Features selecionadas: %d colunas.", features.shape[1])
    return features, alvo


def get_df_preprocessado() -> tuple[pd.DataFrame, pd.Series]:
    """
    Carrega os microdados do INEP e retorna features e alvo para modelagem.

    A variável alvo indica cursos com taxa de evasão acima da mediana,
    calculada a partir da razão entre alunos desvinculados e o total ativo.

    Returns:
        Tupla (X, y) com atributos numéricos e rótulo binário de evasão.

    Raises:
        FileNotFoundError: Se o dataset não for localizado.
        RuntimeError: Se houver falha na leitura do arquivo.
        ValueError: Se os dados não puderem ser preparados.
    """
    try:
        df = _carregar_dados_brutos()
        df = _calcular_alvo(df)
        return _selecionar_features(df)
    except (FileNotFoundError, ValueError):
        raise
    except Exception as exc:
        logger.exception("Erro inesperado ao preprocessar os dados.")
        raise RuntimeError("Falha no pré-processamento dos dados.") from exc
