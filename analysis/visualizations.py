"""Geração de visualizações comparativas entre modelos."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

METRIC_COLUMNS = ["Accuracy", "Precision", "Recall", "F1"]


def _configure_style() -> None:
    """Configura estilo visual consistente para os gráficos."""
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 150,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
    })


def create_metric_heatmap(df: pd.DataFrame, output_path: Path) -> Path:
    """Gera heatmap de métricas por modelo."""
    _configure_style()
    heatmap_data = df.set_index("Modelo")[METRIC_COLUMNS]

    fig, ax = plt.subplots(figsize=(8, max(4, len(df) * 0.8)))
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Valor"},
    )
    ax.set_title("Heatmap de Métricas por Modelo")
    ax.set_xlabel("Métrica")
    ax.set_ylabel("Modelo")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Heatmap de métricas salvo em %s", output_path)
    return output_path


def create_ranking_heatmap(df: pd.DataFrame, output_path: Path) -> Path:
    """Gera heatmap com ranking de cada métrica (1 = melhor)."""
    _configure_style()
    ranking_data = pd.DataFrame(index=df["Modelo"])

    for metric in METRIC_COLUMNS:
        ranking_data[f"Rank {metric}"] = df[metric].rank(
            ascending=False,
            method="min",
        ).astype(int).values

    fig, ax = plt.subplots(figsize=(8, max(4, len(df) * 0.8)))
    sns.heatmap(
        ranking_data,
        annot=True,
        fmt="d",
        cmap="RdYlGn_r",
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Posição (1 = melhor)"},
    )
    ax.set_title("Heatmap de Ranking por Métrica")
    ax.set_xlabel("Ranking")
    ax.set_ylabel("Modelo")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Heatmap de ranking salvo em %s", output_path)
    return output_path


def create_bar_chart(df: pd.DataFrame, output_path: Path) -> Path:
    """Gera gráfico de barras comparando todas as métricas."""
    _configure_style()
    melted = df.melt(
        id_vars="Modelo",
        value_vars=METRIC_COLUMNS,
        var_name="Métrica",
        value_name="Valor",
    )

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(
        data=melted,
        x="Modelo",
        y="Valor",
        hue="Métrica",
        ax=ax,
    )
    ax.set_title("Comparação de Métricas entre Modelos")
    ax.set_xlabel("Modelo")
    ax.set_ylabel("Valor")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=20)
    ax.legend(title="Métrica", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Gráfico de barras salvo em %s", output_path)
    return output_path


def create_radar_chart(df: pd.DataFrame, output_path: Path) -> Path:
    """Gera radar chart com trade-offs entre métricas."""
    _configure_style()
    labels = METRIC_COLUMNS
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})

    for _, row in df.iterrows():
        values = [row[metric] for metric in labels]
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=row["Modelo"])
        ax.fill(angles, values, alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_title("Radar Chart — Trade-offs entre Métricas", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Radar chart salvo em %s", output_path)
    return output_path


def generate_all_visualizations(
    df: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    """Gera todas as visualizações e retorna caminhos dos arquivos."""
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "metric_heatmap": create_metric_heatmap(
            df,
            output_dir / "metric_heatmap.png",
        ),
        "ranking_heatmap": create_ranking_heatmap(
            df,
            output_dir / "ranking_heatmap.png",
        ),
        "bar_chart": create_bar_chart(
            df,
            output_dir / "metrics_bar_chart.png",
        ),
        "radar_chart": create_radar_chart(
            df,
            output_dir / "radar_chart.png",
        ),
    }
    return paths
