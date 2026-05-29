from __future__ import annotations

import importlib


def test_app_imports_without_crashing():
    app = importlib.import_module("app")
    assert hasattr(app, "main")


def test_key_app_functions_importable():
    app = importlib.import_module("app")
    assert callable(app.collect_scenario_watchouts)
    assert callable(app.stable_match_score)
    assert callable(app.build_scenario_inputs)
