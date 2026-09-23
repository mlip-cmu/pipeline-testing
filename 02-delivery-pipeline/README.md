# 02 – From notebook to a tested, automated pipeline

Slides: *Sequential Data Science Code in Notebooks*, *Pipeline restructured into
separate functions*, *Orchestrating Functions*, *Test the Modules*, *Test for
Expected Exceptions*, *Test Model Training Setup*, *Integration tests*,
*End-To-End Test of Entire Pipeline*, *Tracking Model Qualities*.

Predict hourly delivery counts from time and weather (synthetic data in `data/`,
made with `generate_data.py`; the raw data has duplicates, negative counts,
missing values, and impossible humidity values).

| Path | Content |
|---|---|
| `notebooks/delivery_exploration.ipynb` | "Before": sequential notebook code from the slides |
| `delivery/features.py` | "After": cleaning and one function for each feature encoding |
| `delivery/model.py` | `learn`, `evaluate`, `NoDataError` |
| `delivery/pipeline.py` | `prepare_data` and `pipeline` that connect the steps |
| `delivery/stages.py`, `dvc.yaml`, `params.yaml` | The same steps as separate commands, orchestrated with DVC |
| `delivery/tracking.py` | Experiment tracking with MLflow |
| `tests/` | Unit, integration, and end-to-end tests |

## Run

```sh
uv run python -m delivery.pipeline         # train and evaluate once
uv run pytest                              # all tests
```

Tests, from small to large:

* `test_features.py` – unit tests for each encoding (the slide example, boundaries, invalid inputs, zero deliveries for the Box-Cox transform, all categories present even when a subset is given).
* `test_cleaning.py` – data quality checks detect and remove invalid rows.
* `test_model.py` – `NoDataError` for empty data; training setup on a small sample (shapes).
* `test_integration.py` – cleaning plus feature engineering together; train and test data have the same columns.
* `test_pipeline_e2e.py` – full pipeline on small test datasets in `tests/data/`, accuracy threshold.

### Notebook

```sh
uv run --group notebook jupyter lab notebooks/
# run it non-interactively with different parameters (the "parameters" cell is tagged):
cd notebooks && uv run --group notebook papermill delivery_exploration.ipynb ../build/run.ipynb -p test_size 0.2
```

### Orchestration with DVC

`dvc.yaml` declares the stages, their inputs, and outputs. DVC runs only stages
whose inputs changed.

```sh
uv run --group orchestration dvc repro          # run the pipeline
uv run --group orchestration dvc metrics show   # metrics.json
# change train.alpha in params.yaml, then:
uv run --group orchestration dvc repro          # only "train" and "evaluate" run again
uv run --group orchestration dvc params diff
```

### Experiment tracking with MLflow

```sh
uv run --group tracking python -m delivery.tracking --alphas 0 1 10 100
uv run --group tracking mlflow ui --backend-store-uri sqlite:///mlflow.db   # open http://localhost:5000
```
