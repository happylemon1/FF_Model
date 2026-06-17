import subprocess
import sys
import unittest

import pandas as pd

from src import multi_position_pipeline as model_pipeline
from src import rookie_data_pipeline as data_pipeline


WIKI_FIXTURE = """
<html><body>
<table class="wikitable sortable plainrowheaders">
<tr><th></th><th>Rnd.</th><th>Pick</th><th>Team</th><th>Player</th><th>Pos.</th><th>College</th><th>Notes</th></tr>
<tr><td></td><td>1</td><td>1</td><td>LV</td><td>Fernando Mendoza</td><td>QB</td><td>Indiana</td><td></td></tr>
<tr><td></td><td>1</td><td>3</td><td>ARI</td><td>Jeremiyah Love</td><td>RB</td><td>Notre Dame</td><td></td></tr>
<tr><td></td><td>1</td><td>4</td><td>TEN</td><td>Carnell Tate</td><td>WR</td><td>Ohio State</td><td></td></tr>
<tr><td></td><td>1</td><td>16</td><td>CHI</td><td>Kenyon Sadiq</td><td>TE</td><td>Oregon</td><td></td></tr>
<tr><td></td><td>1</td><td>17</td><td>SEA</td><td>Defensive Player</td><td>CB</td><td>Example</td><td></td></tr>
</table>
</body></html>
"""


class MultiPositionPipelineTests(unittest.TestCase):
    def test_wikipedia_draft_parser_filters_skill_positions(self):
        df = data_pipeline.parse_wikipedia_draft_page(WIKI_FIXTURE, 2026)

        self.assertEqual(set(df["position"]), {"QB", "RB", "WR", "TE"})
        self.assertEqual(len(df), 4)
        self.assertIn("college_profile_url", df.columns)
        self.assertTrue(df["college_profile_url"].str.contains("sports-reference.com").all())

    def test_processed_position_datasets_have_features_and_targets(self):
        for position in data_pipeline.POSITIONS:
            df = data_pipeline.build_position_dataset(position, include_targets=True)
            self.assertGreater(len(df), 20)
            for feature in model_pipeline.position_features(position):
                self.assertIn(feature, df.columns)
            for target in model_pipeline.target_columns(position):
                self.assertIn(target, df.columns)
            self.assertFalse(any(column.startswith("predicted_") for column in model_pipeline.position_features(position)))

    def test_2026_feature_files_match_model_features_when_present(self):
        for position in data_pipeline.POSITIONS:
            path = data_pipeline.FINAL_DIR / f"{position.lower()}_rookies_2026_features.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            for feature in model_pipeline.position_features(position):
                self.assertIn(feature, df.columns)

    def test_cli_gpu_info_smoke(self):
        result = subprocess.run(
            [sys.executable, "scripts/rookie_projection.py", "gpu-info"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("nvidia_smi_available", result.stdout)


if __name__ == "__main__":
    unittest.main()
