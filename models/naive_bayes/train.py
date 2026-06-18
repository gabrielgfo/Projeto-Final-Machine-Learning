"""Treinamento comparativo e busca de hiperparâmetros do Naive Bayes."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate import executar_avaliacao
from model_factory import create_model
from preprocessing import (
    aplicar_smote_treino,
    carregar_dados,
    get_preprocessing_pipeline,
    transformar_dados,
)
from report_generator import gerar_relatorio

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
REPORT_DIR = Path(__file__).resolve().parent / "report"

EXPERIMENTS_PER_MODEL = 30
RANDOM_STATE = 42
CV_FOLDS = 5


def _configurar_logging() -> None:
    """Configura logging para execução via linha de comando."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def _garantir_diretorios() -> None:
    """Cria diretórios de saída caso não existam."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def _gerar_configuracoes_aleatorias(rng: np.random.Generator) -> list[dict[str, Any]]:
    """Gera aproximadamente 30 configurações por variante de Naive Bayes."""
    configuracoes: list[dict[str, Any]] = []

    for _ in range(EXPERIMENTS_PER_MODEL):
        configuracoes.append({
            "model_type": "gaussian",
            "var_smoothing": float(rng.choice(np.logspace(-12, -1, 100))),
        })

    for _ in range(EXPERIMENTS_PER_MODEL):
        configuracoes.append({
            "model_type": "bernoulli",
            "alpha": float(rng.choice(np.linspace(0.01, 5.0, 100))),
            "binarize": float(rng.choice(np.linspace(0.0, 1.0, 20))),
        })

    for _ in range(EXPERIMENTS_PER_MODEL):
        configuracoes.append({
            "model_type": "complement",
            "alpha": float(rng.choice(np.linspace(0.01, 5.0, 100))),
            "norm": bool(rng.choice([True, False])),
        })

    rng.shuffle(configuracoes)
    return configuracoes


def _calcular_metricas_fold(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> dict[str, float]:
    """Calcula métricas de classificação para um fold."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }


def _avaliar_configuracao(
    config: dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    skf: StratifiedKFold,
) -> dict[str, Any]:
    """Executa validação cruzada para uma configuração de modelo."""
    model_type = config["model_type"]
    binarize_threshold = float(config.get("binarize", 0.0))

    metricas_folds: list[dict[str, float]] = []

    for train_idx, test_idx in skf.split(X, y):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx].to_numpy()
        y_test = y.iloc[test_idx].to_numpy()

        pipeline = get_preprocessing_pipeline(model_type, binarize_threshold=binarize_threshold)
        X_train_proc, X_test_proc = transformar_dados(
            pipeline,
            X_train,
            X_test,
            y.iloc[train_idx],
            fit=True,
        )
        assert X_test_proc is not None

        X_train_bal, y_train_bal = aplicar_smote_treino(X_train_proc, y_train)

        modelo = create_model(config)
        modelo.fit(X_train_bal, y_train_bal)

        y_pred = modelo.predict(X_test_proc)
        y_proba = modelo.predict_proba(X_test_proc)[:, 1]
        metricas_folds.append(_calcular_metricas_fold(y_test, y_pred, y_proba))

    df_folds = pd.DataFrame(metricas_folds)
    resultado = {
        "model_type": model_type,
        "accuracy": float(df_folds["accuracy"].mean()),
        "precision": float(df_folds["precision"].mean()),
        "recall": float(df_folds["recall"].mean()),
        "f1": float(df_folds["f1"].mean()),
        "roc_auc": float(df_folds["roc_auc"].mean()),
        "recall_std": float(df_folds["recall"].std(ddof=1)),
    }

    for chave, valor in config.items():
        if chave != "model_type":
            resultado[chave] = valor

    return resultado


def _executar_experimentos(
    X: pd.DataFrame,
    y: pd.Series,
) -> pd.DataFrame:
    """Executa todos os experimentos e retorna resultados agregados."""
    rng = np.random.default_rng(RANDOM_STATE)
    configuracoes = _gerar_configuracoes_aleatorias(rng)
    skf = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    resultados: list[dict[str, Any]] = []
    total = len(configuracoes)

    for indice, config in enumerate(configuracoes, start=1):
        logger.info(
            "Experimento %d/%d | modelo=%s | params=%s",
            indice,
            total,
            config["model_type"],
            {k: v for k, v in config.items() if k != "model_type"},
        )
        try:
            resultado = _avaliar_configuracao(config, X, y, skf)
            resultados.append(resultado)
            logger.info(
                "  recall=%.4f | f1=%.4f | roc_auc=%.4f",
                resultado["recall"],
                resultado["f1"],
                resultado["roc_auc"],
            )
        except (ValueError, RuntimeError) as exc:
            logger.warning("Experimento %d ignorado: %s", indice, exc)

    if not resultados:
        raise RuntimeError("Nenhum experimento foi concluído com sucesso.")

    df = pd.DataFrame(resultados)
    caminho = ARTIFACTS_DIR / "all_experiments.csv"
    try:
        df.to_csv(caminho, index=False)
        logger.info("Resultados de todos os experimentos salvos em %s.", caminho)
    except OSError as exc:
        raise RuntimeError(f"Falha ao salvar {caminho}.") from exc

    return df


def _selecionar_melhor_configuracao(df: pd.DataFrame) -> dict[str, Any]:
    """
    Seleciona a melhor configuração por recall médio.

    Critérios de desempate: F1, ROC-AUC e menor desvio padrão do recall.
    """
    ordenado = df.sort_values(
        by=["recall", "f1", "roc_auc", "recall_std"],
        ascending=[False, False, False, True],
    )
    melhor = ordenado.iloc[0].to_dict()

    config: dict[str, Any] = {"model_type": melhor["model_type"]}
    for parametro in ("var_smoothing", "alpha", "binarize", "norm"):
        if parametro in melhor and pd.notna(melhor[parametro]):
            valor = melhor[parametro]
            config[parametro] = bool(valor) if parametro == "norm" else float(valor)

    logger.info(
        "Melhor configuração: %s | recall=%.4f | f1=%.4f | roc_auc=%.4f",
        config,
        melhor["recall"],
        melhor["f1"],
        melhor["roc_auc"],
    )
    return config


def _salvar_melhores_parametros(config: dict[str, Any]) -> None:
    """Persiste os melhores hiperparâmetros em JSON."""
    caminho = ARTIFACTS_DIR / "best_params.json"
    try:
        with caminho.open("w", encoding="utf-8") as arquivo:
            json.dump(config, arquivo, indent=2)
        logger.info("Parâmetros salvos em %s.", caminho)
    except OSError as exc:
        raise RuntimeError(f"Falha ao salvar {caminho}.") from exc


def _salvar_melhor_modelo(
    config: dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
) -> None:
    """Treina e persiste o melhor modelo com pré-processamento no conjunto completo."""
    model_type = config["model_type"]
    binarize_threshold = float(config.get("binarize", 0.0))

    pipeline = get_preprocessing_pipeline(model_type, binarize_threshold=binarize_threshold)
    X_proc, _ = transformar_dados(pipeline, X, y_train=y, fit=True)
    X_train, y_train = aplicar_smote_treino(X_proc, y.to_numpy())

    modelo = create_model(config)
    modelo.fit(X_train, y_train)

    artefato = {
        "config": config,
        "preprocessor": pipeline,
        "model": modelo,
    }

    caminho = ARTIFACTS_DIR / "best_model.pkl"
    try:
        joblib.dump(artefato, caminho)
        logger.info("Modelo salvo em %s.", caminho)
    except OSError as exc:
        raise RuntimeError(f"Falha ao salvar {caminho}.") from exc


def main() -> None:
    """Executa o pipeline completo de treinamento, avaliação e relatório."""
    _configurar_logging()
    _garantir_diretorios()

    try:
        logger.info("Etapa 1/5: carregamento dos dados.")
        X, y = carregar_dados()

        logger.info(
            "Etapa 2/5: busca comparativa (%d experimentos).",
            EXPERIMENTS_PER_MODEL * 3,
        )
        df_experimentos = _executar_experimentos(X, y)
        melhor_config = _selecionar_melhor_configuracao(df_experimentos)

        logger.info("Etapa 3/5: persistência do melhor modelo.")
        _salvar_melhores_parametros(melhor_config)
        _salvar_melhor_modelo(melhor_config, X, y)

        logger.info("Etapa 4/5: avaliação cruzada final.")
        executar_avaliacao(X, y, config=melhor_config)

        logger.info("Etapa 5/5: geração do relatório PDF.")
        gerar_relatorio()

        logger.info("Pipeline concluído com sucesso.")
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logger.error("Pipeline interrompido: %s", exc)
        raise SystemExit(1) from exc
    except Exception as exc:
        logger.exception("Erro inesperado no pipeline.")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
