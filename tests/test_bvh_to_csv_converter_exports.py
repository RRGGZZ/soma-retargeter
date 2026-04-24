from __future__ import annotations

import pytest

from app.bvh_to_csv_converter import _resolve_export_settings


def test_resolve_export_settings_uses_json_path_for_pkl_only() -> None:
    export_path, export_pkl_only = _resolve_export_settings(
        {"export_folder": "/tmp/custom/adam_soma_pkl"}
    )

    assert str(export_path) == "/tmp/custom/adam_soma_pkl"
    assert export_pkl_only is True


def test_resolve_export_settings_allows_non_pkl_folder() -> None:
    export_path, export_pkl_only = _resolve_export_settings(
        {"export_folder": "/tmp/custom/adam_csv_and_joblib"}
    )

    assert str(export_path) == "/tmp/custom/adam_csv_and_joblib"
    assert export_pkl_only is False


def test_resolve_export_settings_requires_export_folder() -> None:
    with pytest.raises(ValueError, match="No export folder specified"):
        _resolve_export_settings({})
