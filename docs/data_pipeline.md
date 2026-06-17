# Data Pipeline

## Main Commands

```powershell
python scripts\rookie_projection.py build-data
python scripts\rookie_projection.py crawl --year 2026
```

## Sources

- Historical training data is built from existing repo CSVs in `data/raw/` for QB, RB, WR, and TE.
- The 2026 rookie draft list is fetched from Pro Football Reference when available.
- If Pro Football Reference blocks scripted access, the crawler falls back to the Wikipedia 2026 NFL Draft table.
- Sports Reference college profile URLs are attempted for rookie college stat refreshes. If blocked, the pipeline writes draft-only features and logs the scrape error in the raw rookie college CSVs.

## Outputs

- Historical processed datasets:
  - `data/processed/qb_training_dataset.csv`
  - `data/processed/rb_training_dataset.csv`
  - `data/processed/wr_training_dataset.csv`
  - `data/processed/te_training_dataset.csv`
  - `data/processed/all_positions_training_dataset.csv`
- 2026 rookie inputs:
  - `data/raw/rookies_2026_draft.csv`
  - `data/final/qb_rookies_2026_features.csv`
  - `data/final/rb_rookies_2026_features.csv`
  - `data/final/wr_rookies_2026_features.csv`
  - `data/final/te_rookies_2026_features.csv`

## Notes

The current run encountered HTTP 403 responses from Pro Football Reference and Sports Reference, so 2026 inference uses draft capital, position, school/conference, and zero-filled production features. This is enough to run the model stack, but future refreshes should prefer full college-stat features when the source sites allow access.
