"""LightGBM experiment entry-point.

This script is retained for backwards compatibility and now proxies to the
config-driven experiment framework.  It simply resolves the default
configuration file and delegates to :mod:`exp.ml_pipeline`.
"""

from pathlib import Path

from .ml_pipeline import format_results, run_experiment


def main() -> None:
    config_path = Path(__file__).resolve().parent / "config" / "ml_config.json"
    outcome = run_experiment(config_path)
    print(format_results(outcome["results"]))


if __name__ == "__main__":
    main()
