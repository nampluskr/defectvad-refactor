import csv
import json
import os
import tempfile
import unittest

from src.batch.parser import BatchCase, expand_batch_config
from src.batch.runner import BatchRunner
from src.batch.summary import (
    BatchSummary,
    CaseResult,
    compute_mean_metrics,
    render_summary_table,
    save_summary_csv,
    save_summary_json,
)
from src.utils.io import save_yaml


class TestBatchSystem(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_expand_batch_grid(self):
        matrix_yaml = {
            "meta": {"name": "test_grid", "task": "anomaly"},
            "base_data": "configs/anomaly/data/mvtec.yaml",
            "base_model": "configs/anomaly/models/stfpm.yaml",
            "matrix": {
                "data": {"category": ["bottle", "cable", "capsule"]},
                "model": {"backbone": ["resnet18", "resnet50"]},
            },
            "execution": {
                "run_name_pattern": "test_{data_category}_{model_backbone}",
                "output_root": os.path.join(self.temp_dir, "outputs"),
            },
        }
        cfg_path = os.path.join(self.temp_dir, "grid.yaml")
        save_yaml(matrix_yaml, cfg_path)

        res = expand_batch_config(cfg_path)
        self.assertEqual(res.name, "test_grid")
        self.assertEqual(len(res.cases), 6)  # 3 categories x 2 backbones

        case0 = res.cases[0]
        self.assertEqual(case0.data_selectors, {"category": "bottle"})
        self.assertEqual(case0.model_selectors, {"backbone": "resnet18"})
        self.assertEqual(case0.run_name, "test_bottle_resnet18")
        self.assertTrue(case0.output_dir.endswith("test_bottle_resnet18"))

    def test_expand_batch_cases_list(self):
        cases_yaml = {
            "meta": {"name": "test_cases", "task": "anomaly"},
            "base_data": "configs/anomaly/data/mvtec.yaml",
            "base_model": "configs/anomaly/models/stfpm.yaml",
            "cases": [
                {
                    "name": "case_bottle",
                    "data_selectors": {"category": "bottle"},
                    "model_selectors": {"backbone": "resnet18"},
                    "overrides": ["train.epochs=10"],
                },
                {
                    "name": "case_capsule_effad",
                    "model": "configs/anomaly/models/efficientad.yaml",
                    "data_selectors": {"category": "capsule"},
                    "model_selectors": {"size": "small"},
                    "overrides": ["train.epochs=20"],
                },
            ],
            "execution": {
                "output_root": os.path.join(self.temp_dir, "outputs_cases"),
            },
        }
        cfg_path = os.path.join(self.temp_dir, "cases.yaml")
        save_yaml(cases_yaml, cfg_path)

        res = expand_batch_config(cfg_path)
        self.assertEqual(res.name, "test_cases")
        self.assertEqual(len(res.cases), 2)

        c0 = res.cases[0]
        self.assertEqual(c0.case_id, "case_bottle")
        self.assertEqual(c0.model_path, "configs/anomaly/models/stfpm.yaml")
        self.assertIn("train.epochs=10", c0.extra_overrides)

        c1 = res.cases[1]
        self.assertEqual(c1.case_id, "case_capsule_effad")
        self.assertEqual(c1.model_path, "configs/anomaly/models/efficientad.yaml")
        self.assertIn("train.epochs=20", c1.extra_overrides)

    def test_expand_batch_filter_only(self):
        matrix_yaml = {
            "meta": {"name": "test_filter", "task": "anomaly"},
            "base_data": "configs/anomaly/data/mvtec.yaml",
            "base_model": "configs/anomaly/models/stfpm.yaml",
            "matrix": {
                "data": {"category": ["bottle", "cable", "carpet"]},
                "model": {"backbone": ["resnet18"]},
            },
        }
        res = expand_batch_config(matrix_yaml, only="bottle,carpet")
        self.assertEqual(len(res.cases), 2)
        categories = [c.data_selectors["category"] for c in res.cases]
        self.assertIn("bottle", categories)
        self.assertIn("carpet", categories)
        self.assertNotIn("cable", categories)

    def test_summary_and_reporting(self):
        results = [
            CaseResult(
                case_id="bottle",
                run_name="stfpm_resnet18_bottle",
                status="SUCCESS",
                mode="evaluate",
                metrics={"image_auroc": 0.9850, "pixel_auroc": 0.9720},
                elapsed_sec=12.5,
                output_dir=os.path.join(self.temp_dir, "bottle"),
                meta={"category": "bottle", "backbone": "resnet18"},
            ),
            CaseResult(
                case_id="cable",
                run_name="stfpm_resnet18_cable",
                status="SUCCESS",
                mode="evaluate",
                metrics={"image_auroc": 0.9250, "pixel_auroc": 0.9600},
                elapsed_sec=14.2,
                output_dir=os.path.join(self.temp_dir, "cable"),
                meta={"category": "cable", "backbone": "resnet18"},
            ),
            CaseResult(
                case_id="capsule",
                run_name="stfpm_resnet18_capsule",
                status="FAILED",
                mode="evaluate",
                metrics={},
                elapsed_sec=0.5,
                error_msg="Checkpoint file missing",
                output_dir=os.path.join(self.temp_dir, "capsule"),
                meta={"category": "capsule", "backbone": "resnet18"},
            ),
        ]

        means = compute_mean_metrics(results)
        self.assertAlmostEqual(means["image_auroc"], (0.9850 + 0.9250) / 2, places=4)
        self.assertAlmostEqual(means["pixel_auroc"], (0.9720 + 0.9600) / 2, places=4)

        summary = BatchSummary(
            batch_name="test_summary",
            task_name="anomaly",
            mode="evaluate",
            total_cases=3,
            success_cases=2,
            failed_cases=1,
            skipped_cases=0,
            total_elapsed_sec=27.2,
            results=results,
            mean_metrics=means,
        )

        # Test JSON export
        json_path = os.path.join(self.temp_dir, "summary.json")
        save_summary_json(summary, json_path)
        self.assertTrue(os.path.isfile(json_path))

        # Test CSV export
        csv_path = os.path.join(self.temp_dir, "summary.csv")
        save_summary_csv(summary, csv_path)
        self.assertTrue(os.path.isfile(csv_path))

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            self.assertEqual(len(rows), 4)  # 3 cases + 1 MEAN summary row
            self.assertEqual(rows[-1]["case_id"], "MEAN")

        # Test ASCII table render
        table = render_summary_table(summary)
        self.assertIn("Image Auroc", table)
        self.assertIn("Pixel Auroc", table)
        self.assertIn("MEAN", table)
        self.assertIn("bottle", table)

    def test_fault_tolerant_runner_mock(self):
        class MockBatchRunner(BatchRunner):
            def _execute_case(self, case: BatchCase):
                if case.case_id == "failing_case":
                    raise RuntimeError("Simulated failure")
                return {"image_auroc": 0.99}

        from src.batch.parser import BatchConfig

        cases = [
            BatchCase(
                case_id="success_case",
                data_path="configs/anomaly/data/mvtec.yaml",
                model_path="configs/anomaly/models/stfpm.yaml",
                run_name="run_success",
                output_dir=os.path.join(self.temp_dir, "run_success"),
            ),
            BatchCase(
                case_id="failing_case",
                data_path="configs/anomaly/data/mvtec.yaml",
                model_path="configs/anomaly/models/stfpm.yaml",
                run_name="run_fail",
                output_dir=os.path.join(self.temp_dir, "run_fail"),
            ),
        ]
        batch_cfg = BatchConfig(
            name="mock_batch",
            task_name="anomaly",
            base_data="configs/anomaly/data/mvtec.yaml",
            base_model="configs/anomaly/models/stfpm.yaml",
            cases=cases,
            output_root=self.temp_dir,
        )

        runner = MockBatchRunner(batch_cfg, mode="evaluate")
        summary = runner.run()

        self.assertEqual(summary.total_cases, 2)
        self.assertEqual(summary.success_cases, 1)
        self.assertEqual(summary.failed_cases, 1)
        self.assertEqual(summary.results[0].status, "SUCCESS")
        self.assertEqual(summary.results[1].status, "FAILED")
        self.assertEqual(summary.results[1].error_msg, "Simulated failure")


if __name__ == "__main__":
    unittest.main()
