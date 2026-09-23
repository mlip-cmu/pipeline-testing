# 04 – Testing across boundaries: stubbing dependencies

Slides: *How to unit test component with dependency on other code?*,
*Decoupling from Dependencies*, *Stubbing the Dependency*,
*General Testing Strategy: Decoupling Code Under Test*.

`clean_gender` fills in missing genders with an external web API
([genderize.io](https://genderize.io)). Tests that call the real API are slow,
flaky, rate limited, and not reproducible.

| File | Content |
|---|---|
| `gender/api_client.py` | Real HTTP client for the API |
| `gender/cleaning_hardcoded.py` | Original version: uses a global API client |
| `gender/cleaning.py` | Decoupled version: the caller passes in the model (any function) |
| `tests/test_cleaning.py` | Tests with a hand-written stub (slide) and with `unittest.mock.Mock` |
| `tests/test_cleaning_hardcoded.py` | The original is testable only by patching the module |
| `tests/test_api_client.py` | The client itself, tested with a stubbed HTTP transport |
| `demo.py` | Runs the cleaning with the real API |

## Run

```sh
uv run pytest            # no network access needed
uv run python demo.py    # calls the real API
```
