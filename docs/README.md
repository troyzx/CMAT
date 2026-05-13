# Documentation

CMAT's user-facing documentation is split into focused entry points:

- [Installation](INSTALLATION.md)
- [Quick Start](QUICKSTART.md)
- [Theory](THEORY.md)
- [API Reference](API_REFERENCE.md)
- [Data Formats](DATA_FORMATS.md)
- [Benchmarking](BENCHMARKING.md)
- [Troubleshooting](TROUBLESHOOTING.md)

The rebuild-specific engineering notes remain under [`docs/rebuild/`](rebuild/).

The theory guide now covers both the astronomy-specific interpretation and the more general inverse-modeling pattern behind the workflow.

For small self-contained examples, see:

- [`examples/synthetic_ttv_quickstart.py`](../examples/synthetic_ttv_quickstart.py)
- [`examples/synthetic_ttv_quickstart.ipynb`](../examples/synthetic_ttv_quickstart.ipynb)
- [`examples/custom_scoring_backend.py`](../examples/custom_scoring_backend.py)
- [`examples/damped_oscillator_adapter.py`](../examples/damped_oscillator_adapter.py)
- [`examples/damped_oscillator_batch.py`](../examples/damped_oscillator_batch.py)

The damped-oscillator adapter example is the current Stage 5 portfolio-output demo: it writes a report, score table, summary JSON, and static figure bundle. The batch variant adds a deployment-style run folder with `run_metadata.json` for reproducible reruns and artifact storage.
