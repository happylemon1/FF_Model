# Modeling Report

Generated: 2026-06-17T21:12:49.272157+00:00

## GPU Status

```json
{
  "nvidia_smi_available": true,
  "nvidia_gpu": "NVIDIA RTX A1000 6GB Laptop GPU, 6144 MiB",
  "torch_available": true,
  "torch_cuda_available": false,
  "torch_device_count": 0,
  "torch_version": "2.12.0+cpu",
  "xgboost_available": true,
  "xgboost_version": "3.2.0"
}
```

## Selected Models

### QB
- `fantasy_points`: `elastic_net` (linear), MAE=54.096, Spearman=0.678
- `nfl_passingINT`: `ridge` (linear), MAE=2.831, Spearman=0.464
- `nfl_passingTDS`: `random_forest` (bagging_tree), MAE=5.779, Spearman=0.203
- `nfl_passingYDS`: `ridge` (linear), MAE=736.062, Spearman=0.777
- `nfl_rushingTDS`: `extra_trees` (bagging_tree), MAE=1.042, Spearman=0.571
- `nfl_rushingYDS`: `mlp` (neural_net), MAE=119.638, Spearman=0.850

### RB
- `fantasy_points`: `extra_trees` (bagging_tree), MAE=43.366, Spearman=0.659
- `nfl_recTDS`: `median` (baseline), MAE=0.539, Spearman=nan
- `nfl_recYDS`: `ridge` (linear), MAE=74.042, Spearman=0.613
- `nfl_recs`: `extra_trees` (bagging_tree), MAE=9.519, Spearman=0.562
- `nfl_rushingTDS`: `median` (baseline), MAE=1.761, Spearman=nan
- `nfl_rushingYDS`: `random_forest` (bagging_tree), MAE=180.235, Spearman=0.568

### WR
- `fantasy_points`: `extra_trees` (bagging_tree), MAE=44.775, Spearman=0.658
- `nfl_recTDS`: `extra_trees` (bagging_tree), MAE=1.910, Spearman=0.486
- `nfl_recYDS`: `residual_mlp` (neural_net), MAE=205.492, Spearman=0.627
- `nfl_recs`: `residual_mlp` (neural_net), MAE=15.089, Spearman=0.661

### TE
- `fantasy_points`: `gradient_boosting` (boosted_tree), MAE=32.917, Spearman=0.677
- `nfl_recTDS`: `random_forest` (bagging_tree), MAE=1.095, Spearman=0.632
- `nfl_recYDS`: `extra_trees` (bagging_tree), MAE=136.302, Spearman=0.543
- `nfl_recs`: `extra_trees` (bagging_tree), MAE=13.655, Spearman=0.632

### ALL
- `fantasy_points`: `extra_trees` (bagging_tree), MAE=45.010, Spearman=0.634

## Outputs

- Artifacts: `C:\Users\danie\Repositories\FF_Model\models\artifacts\multi_position`
- Leaderboard: `C:\Users\danie\Repositories\FF_Model\models\artifacts\multi_position\leaderboard.csv`
- Combined predictions: `C:\Users\danie\Repositories\FF_Model\data\final\rookie_predictions_2026_combined.csv`

## Caveats

- Historical data is small by modern ML standards, so tree ensembles and regularized linear models may beat neural nets.
- 2026 rookie inference quality depends on the availability and shape of Sports Reference college profile tables.
- GPU acceleration is attempted for XGBoost when available; PyTorch CUDA requires a CUDA-enabled PyTorch wheel.
