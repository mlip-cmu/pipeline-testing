import json
from pathlib import Path

from delivery import stages
from delivery.pipeline import pipeline

DATA = Path(__file__).parent / "data"


def test_pipeline():
    model, accuracy = pipeline(DATA / "pipelinetest_training.csv", DATA / "pipelinetest_test.csv")
    assert accuracy > 0.9


def test_pipeline_stages(tmp_path):
    stages.prepare(DATA / "pipelinetest_training.csv", tmp_path / "train.csv")
    stages.prepare(DATA / "pipelinetest_test.csv", tmp_path / "test.csv")
    stages.train(tmp_path / "train.csv", tmp_path / "model.joblib", alpha=1.0)
    stages.evaluate_model(tmp_path / "model.joblib", tmp_path / "test.csv", tmp_path / "metrics.json")
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["r2"] > 0.9
    assert metrics["mae_deliveries"] < 15
