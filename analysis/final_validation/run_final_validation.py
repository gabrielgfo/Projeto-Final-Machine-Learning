"""Orquestrador da auditoria científica final (fases 1–13)."""

from __future__ import annotations

import json
import logging
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.final_validation.consistency import (
    build_extracted_metrics,
    build_inconsistencies_report,
    build_project_inventory,
    validate_metrics_from_folds,
)
from analysis.final_validation.decision_tree_audit import run_decision_tree_audit
from analysis.final_validation.latex_report import generate_latex_report
from analysis.final_validation.plots import generate_all_validation_plots
from analysis.final_validation.ranking import build_validated_ranking, build_ranking_analysis
from analysis.final_validation.scientific_discussion import build_scientific_discussion

logger = logging.getLogger(__name__)

RESULTS = PROJECT_ROOT / "results"
VALIDATION = RESULTS / "validation"
DECISION_TREE_DIR = RESULTS / "decision_tree"
FEATURE_DIR = RESULTS / "feature_importance"


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _ensure_dirs() -> None:
    for d in (VALIDATION, DECISION_TREE_DIR, FEATURE_DIR):
        d.mkdir(parents=True, exist_ok=True)


def main() -> None:
    _configure_logging()
    _ensure_dirs()
    today = date.today().isoformat()

    logger.info("Fase 1: inventário do projeto")
    inv_path = build_project_inventory(RESULTS / "project_inventory.md", today)

    logger.info("Fase 2: extração de métricas")
    ext_path = build_extracted_metrics(RESULTS / "extracted_metrics.csv")

    logger.info("Fase 3: relatório de inconsistências")
    inc_path = build_inconsistencies_report(RESULTS / "inconsistencies_report.md")

    logger.info("Fases 4–8: validação de métricas e visualizações")
    val_path = validate_metrics_from_folds(RESULTS / "validated_metrics.csv")
    plot_paths = generate_all_validation_plots(VALIDATION)

    logger.info("Fase 9: ranking validado (pesos iguais 20%%)")
    rank_path = build_validated_ranking(RESULTS / "final_validated_ranking.csv")
    rank_analysis_path = build_ranking_analysis(RESULTS / "ranking_analysis.md")

    logger.info("Fases 10–11: auditoria da árvore de decisão e importância")
    dt_paths = run_decision_tree_audit(DECISION_TREE_DIR, FEATURE_DIR)

    logger.info("Fase 12: discussão científica")
    sci_path = build_scientific_discussion(RESULTS / "scientific_discussion.md")

    logger.info("Fase 13: relatório LaTeX")
    tex_path = generate_latex_report(RESULTS / "final_report.tex", today)

    summary = {
        "date": today,
        "outputs": {
            "inventory": str(inv_path),
            "extracted_metrics": str(ext_path),
            "inconsistencies": str(inc_path),
            "validated_metrics": str(val_path),
            "validation_plots": [str(p) for p in plot_paths],
            "final_validated_ranking": str(rank_path),
            "ranking_analysis": str(rank_analysis_path),
            "decision_tree": {k: str(v) for k, v in dt_paths.items()},
            "scientific_discussion": str(sci_path),
            "latex_report": str(tex_path),
        },
    }
    manifest = RESULTS / "final_validation_manifest.json"
    manifest.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Auditoria final concluída. Manifesto: %s", manifest)


if __name__ == "__main__":
    main()
