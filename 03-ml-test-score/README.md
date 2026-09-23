# 03 – ML Test Score for a Covid-19 detection model

Slides: *Data Tests*, *Tests for Model Development*, *ML Infrastructure Tests*,
*Monitoring Tests*, *Case Study: Covid-19 Detection*.

Scenario: a smartphone app records a cough and sends it to the cloud. A model
predicts whether the user has Covid-19. The recordings are simulated
(`covid/recordings.py`): cough bursts with a different spectrum for positive
cases, filtered by the microphone of three phone models, plus metadata (age,
sex, fever) and personal data that must not be used (user ID, phone number,
location).

| Module | Content |
|---|---|
| `covid/recordings.py` | Simulated app uploads |
| `covid/features.py` | Audio feature extraction, used for training *and* serving |
| `covid/schema.py` | Expected ranges and values for all features |
| `covid/model.py` | `ModelSpec`, training, evaluation (AUC, recall) |
| `covid/registry.py` | Model registry: quality gate, canary, rollback, metadata |
| `covid/serving.py` | Prediction service with input checks and explanations |
| `covid/monitoring.py` | Production checks that send notifications |
| `covid/train.py` | Command: simulate data, train, validate, deploy |

Tests follow the four groups of the rubric (Breck et al., 2017):

| Rubric item | Test |
|---|---|
| **Data** | `tests/test_data.py` |
| Feature expectations are captured in a schema | `test_feature_expectations_are_captured_in_schema`, `test_schema_detects_problems` |
| All features are beneficial | `test_no_feature_harms_model_quality` (ablation, one test per feature) |
| No feature's cost is too much | `test_feature_cost_is_acceptable` |
| Meta-level requirements / privacy controls | `test_no_personal_data_in_features`, `test_personal_data_cannot_be_used_as_feature` |
| All input feature code is tested | `test_spectral_centroid_of_pure_tone`, `test_count_coughs_...`, `test_feature_extraction_handles_degenerate_audio` |
| **Model development** | `tests/test_model_development.py` |
| A simpler model is not better | `test_simpler_model_is_not_better` (age + fever only) |
| All hyperparameters have been tuned | `test_hyperparameters_are_tuned` |
| Model quality is sufficient on important data slices | `test_quality_on_phone_model_slices` |
| The model is tested for considerations of inclusion | `test_recall_similar_across_sexes`, `..._age_groups` |
| **Infrastructure** | `tests/test_infrastructure.py` |
| Training is reproducible | `test_training_is_reproducible` |
| Model specs are unit tested | `test_invalid_model_specs_are_rejected` |
| The ML pipeline is integration tested | `test_pipeline_integration` |
| Model quality is validated before serving | `test_model_quality_is_validated_before_serving` |
| The model is debuggable | `test_model_is_debuggable` |
| Models are canaried before serving | `test_canary_prevents_deploying_broken_model` |
| Serving models can be rolled back | `test_serving_model_can_be_rolled_back` |
| **Monitoring** | `tests/test_monitoring.py` |
| Dependency changes result in notification | `test_dependency_changes_result_in_notification` |
| Data invariants hold for inputs | `test_data_invariants_hold_for_inputs`, `test_input_monitor_alerts_...` |
| Training and serving are not skewed | `test_training_and_serving_features_are_identical`, `test_training_serving_skew_is_detected` |
| Models are not too stale | `test_models_are_not_too_stale` |
| Models are numerically stable | `test_model_is_numerically_stable` |
| Computing performance has not regressed | `test_computing_performance_has_not_regressed` |
| Prediction quality has not regressed | `test_prediction_quality_regression_is_detected` |

Monitoring tests use a `FakeNotifier` (see `tests/conftest.py`) instead of a real
pager or chat service and assert that a notification was sent.

## Run

```sh
uv run pytest                                  # all tests (about 20 s)
uv run pytest tests/test_data.py -v            # one group
uv run python -m covid.train                   # train and deploy to build/registry
uv run python -m covid.train --seed 3          # a second model; deployed only if not worse
uv run python -m covid.train --C 0.0001        # rejected by the quality gate
```

Some items of the rubric are about process (specs are reviewed, offline and
online metrics correlate, new features can be added quickly) and have no test here.
