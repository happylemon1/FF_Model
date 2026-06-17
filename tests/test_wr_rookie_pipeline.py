import unittest

import numpy as np
import pandas as pd

from src import wr_rookie_pipeline as pipeline


class WrRookiePipelineTests(unittest.TestCase):
    def test_training_data_loads_without_target_leakage(self):
        x, y, meta = pipeline.prepare_training_data()

        self.assertGreater(len(x), 0)
        self.assertEqual(list(x.columns), pipeline.FEATURE_COLUMNS)
        self.assertEqual(list(y.columns), pipeline.TARGET_COLUMNS)
        self.assertFalse(any(column.startswith("nfl_") for column in x.columns))
        self.assertTrue((meta["nfl_games"] >= pipeline.MIN_NFL_GAMES).all())

    def test_rookie_features_match_training_features(self):
        x_train, _, _ = pipeline.prepare_training_data()
        x_rookie, rookie_meta = pipeline.prepare_rookie_data(
            feature_columns=pipeline.FEATURE_COLUMNS
        )

        self.assertEqual(list(x_rookie.columns), list(x_train.columns))
        self.assertGreater(len(rookie_meta), 0)
        self.assertIn("name", rookie_meta.columns)

    def test_time_splits_are_ordered(self):
        _, _, meta = pipeline.prepare_training_data()
        splits = pipeline.make_time_splits(meta, max_folds=3)

        self.assertGreaterEqual(len(splits), 1)
        seasons = pd.to_numeric(meta["season"], errors="coerce").to_numpy()
        for train_idx, valid_idx in splits:
            self.assertGreater(len(train_idx), 0)
            self.assertGreater(len(valid_idx), 0)
            self.assertLess(seasons[train_idx].max(), seasons[valid_idx].min())

    def test_rank_score_is_finite(self):
        df = pd.DataFrame(
            {
                "predicted_nfl_YPG": [10.0, 30.0, 20.0],
                "predicted_nfl_RPG": [1.0, 3.0, 2.0],
                "predicted_nfl_TDPG": [0.1, 0.2, 0.3],
            }
        )
        score = pipeline.compute_rank_score(df)

        self.assertEqual(len(score), len(df))
        self.assertTrue(np.isfinite(score).all())
        self.assertEqual(int(np.argmax(score)), 1)


if __name__ == "__main__":
    unittest.main()
