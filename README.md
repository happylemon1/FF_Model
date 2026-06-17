train_receiver_model.py is the legacy driver code for rookie_receiver.py.
Get conferences using rookie_college_receiver.py.

## WR rookie projection pipeline

The upgraded WR pipeline trains tabular ensemble and neural-network candidates,
selects models by time-aware validation, and writes 2025 rookie projections.

### Install

```powershell
python -m pip install -r requirements.txt
```

For NVIDIA GPU training, reinstall PyTorch from the official CUDA wheel index
that matches your Python version and driver. Example:

```powershell
python -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu128
```

### Commands

```powershell
python scripts\wr_rookie_model.py gpu-info
python scripts\wr_rookie_model.py smoke
python scripts\wr_rookie_model.py train --trials 6 --include-neural
python scripts\wr_rookie_model.py evaluate
python scripts\wr_rookie_model.py predict
```

Artifacts are saved under `models/artifacts/`, and rookie inference writes
`models/rookie_predictions.csv`.

## Multi-position rookie projection pipeline

The newer pipeline generalizes the workflow across QB, RB, WR, and TE and
targets the 2026 rookie class.

```powershell
python scripts\rookie_projection.py gpu-info
python scripts\rookie_projection.py build-data
python scripts\rookie_projection.py crawl --year 2026
python scripts\rookie_projection.py train --folds 2 --include-neural
python scripts\rookie_projection.py evaluate
python scripts\rookie_projection.py predict --year 2026
python scripts\rookie_projection.py report --year 2026
```

The main outputs are:

- `models/artifacts/multi_position/leaderboard.csv`
- `data/final/rookie_predictions_2026_combined.csv`
- `docs/modeling_report.md`
- `docs/data_pipeline.md`
- `docs/artifact_index.md`

During the current run, Pro Football Reference and Sports Reference returned
HTTP 403 responses to scripted requests, so the crawler used Wikipedia for the
2026 draft list and generated draft-only inference features.
