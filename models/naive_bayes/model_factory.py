"""Fábrica de estimadores Naive Bayes."""

from __future__ import annotations

from typing import Any

from sklearn.naive_bayes import BernoulliNB, ComplementNB, GaussianNB


def create_model(config: dict[str, Any]) -> GaussianNB | BernoulliNB | ComplementNB:
    """
    Cria o estimador Naive Bayes conforme a configuração informada.

    Args:
        config: Dicionário com model_type e hiperparâmetros específicos.

    Returns:
        Estimador sklearn correspondente à variante solicitada.

    Raises:
        ValueError: Se model_type for inválido ou parâmetros estiverem ausentes.
    """
    model_type = config.get("model_type")

    if model_type == "gaussian":
        if "var_smoothing" not in config:
            raise ValueError("Configuração gaussian requer 'var_smoothing'.")
        return GaussianNB(var_smoothing=float(config["var_smoothing"]))

    if model_type == "bernoulli":
        if "alpha" not in config or "binarize" not in config:
            raise ValueError("Configuração bernoulli requer 'alpha' e 'binarize'.")
        return BernoulliNB(
            alpha=float(config["alpha"]),
            binarize=float(config["binarize"]),
        )

    if model_type == "complement":
        if "alpha" not in config or "norm" not in config:
            raise ValueError("Configuração complement requer 'alpha' e 'norm'.")
        return ComplementNB(
            alpha=float(config["alpha"]),
            norm=bool(config["norm"]),
        )

    raise ValueError(
        f"model_type inválido: {model_type!r}. "
        "Use 'gaussian', 'bernoulli' ou 'complement'."
    )
