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
