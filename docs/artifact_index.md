# Artifact Index

## Model Artifacts

- WR-specific first-pass artifacts: `models/artifacts/`
- Multi-position artifacts: `models/artifacts/multi_position/`
- Multi-position leaderboard: `models/artifacts/multi_position/leaderboard.csv`
- Multi-position metadata: `models/artifacts/multi_position/metadata.json`

## Prediction Outputs

- Combined 2026 rookie ranking: `data/final/rookie_predictions_2026_combined.csv`
- Position-specific 2026 predictions:
  - `data/final/qb_rookie_predictions_2026.csv`
  - `data/final/rb_rookie_predictions_2026.csv`
  - `data/final/wr_rookie_predictions_2026.csv`
  - `data/final/te_rookie_predictions_2026.csv`

## Reports

- Human-readable modeling report: `docs/modeling_report.md`
- Timestamped/generated report copy: `reports/modeling_report_2026.md`
- Data collection summary: `reports/data_collection_2026.json`

## Reproduction

```powershell
python scripts\rookie_projection.py build-data
python scripts\rookie_projection.py crawl --year 2026
python scripts\rookie_projection.py train --folds 2 --include-neural
python scripts\rookie_projection.py predict --year 2026
python scripts\rookie_projection.py report --year 2026
```
