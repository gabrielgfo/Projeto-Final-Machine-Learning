"""Script principal do pipeline de análise comparativa entre modelos."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.metrics_discovery import (
    RESULTS_DIR,
    compute_ranking,
    discover_model_metrics,
    save_consolidated_outputs,
)
from analysis.report_generator import generate_final_report
from analysis.scientific_analysis import build_analysis_bundle
from analysis.visualizations import generate_all_visualizations

logger = logging.getLogger(__name__)

VISUALIZATIONS_DIR = RESULTS_DIR / "visualizations"
REPORT_PATH = RESULTS_DIR / "final_report.pdf"


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def main() -> None:
    """Executa o pipeline completo de comparação entre modelos."""
    _configure_logging()

    try:
        logger.info("Etapa 1/5: descoberta automática de métricas em models/.")
        comparison_df = discover_model_metrics()

        logger.info("Etapa 2/5: cálculo do ranking global.")
        ranking_df = compute_ranking(comparison_df.drop(columns=["fonte"], errors="ignore"))
        save_consolidated_outputs(comparison_df, ranking_df)

        logger.info("Etapa 3/5: geração de visualizações.")
        visualization_paths = generate_all_visualizations(
            comparison_df.drop(columns=["fonte"], errors="ignore"),
            VISUALIZATIONS_DIR,
        )

        logger.info("Etapa 4/5: análise científica automatizada.")
        analysis_bundle = build_analysis_bundle(
            comparison_df.drop(columns=["fonte"], errors="ignore"),
            ranking_df,
        )

        logger.info("Etapa 5/5: geração do relatório PDF.")
        generate_final_report(
            comparison_df.drop(columns=["fonte"], errors="ignore"),
            ranking_df,
            analysis_bundle,
            visualization_paths,
            REPORT_PATH,
        )

        logger.info("Pipeline comparativo concluído com sucesso.")
        logger.info("Arquivos gerados em: %s", RESULTS_DIR)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        logger.error("Pipeline interrompido: %s", exc)
        raise SystemExit(1) from exc
    except Exception as exc:
        logger.exception("Erro inesperado no pipeline comparativo.")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
