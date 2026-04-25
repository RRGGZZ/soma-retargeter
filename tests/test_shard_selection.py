from __future__ import annotations

import pytest

from app.bvh_to_csv_converter import _apply_shard_filter


def _files(n: int) -> list[str]:
    return [f"file_{i}.bvh" for i in range(n)]


def test_no_shard_keys_returns_full_list() -> None:
    files = _files(10)
    assert _apply_shard_filter(files, {}) == files


def test_shard_count_one_returns_full_list() -> None:
    files = _files(10)
    assert _apply_shard_filter(files, {"shard_index": 0, "shard_count": 1}) == files


def test_shard0_gets_even_indices() -> None:
    files = _files(6)
    result = _apply_shard_filter(files, {"shard_index": 0, "shard_count": 2})
    assert result == [files[0], files[2], files[4]]


def test_shard1_gets_odd_indices() -> None:
    files = _files(6)
    result = _apply_shard_filter(files, {"shard_index": 1, "shard_count": 2})
    assert result == [files[1], files[3], files[5]]


def test_disjoint_union_equals_full_list() -> None:
    files = _files(10)
    shard0 = _apply_shard_filter(files, {"shard_index": 0, "shard_count": 2})
    shard1 = _apply_shard_filter(files, {"shard_index": 1, "shard_count": 2})
    assert set(shard0) & set(shard1) == set()
    assert set(shard0) | set(shard1) == set(files)


def test_odd_file_count_shard0_gets_more() -> None:
    files = _files(7)
    shard0 = _apply_shard_filter(files, {"shard_index": 0, "shard_count": 2})
    shard1 = _apply_shard_filter(files, {"shard_index": 1, "shard_count": 2})
    assert len(shard0) == 4
    assert len(shard1) == 3
    assert set(shard0) | set(shard1) == set(files)


def test_single_file_shard1_is_empty() -> None:
    files = _files(1)
    result = _apply_shard_filter(files, {"shard_index": 1, "shard_count": 2})
    assert result == []


def test_single_file_shard0_gets_it() -> None:
    files = _files(1)
    result = _apply_shard_filter(files, {"shard_index": 0, "shard_count": 2})
    assert result == files


def test_missing_shard_index_returns_full_list() -> None:
    files = _files(5)
    assert _apply_shard_filter(files, {"shard_count": 2}) == files


def test_missing_shard_count_returns_full_list() -> None:
    files = _files(5)
    assert _apply_shard_filter(files, {"shard_index": 0}) == files


def test_order_preserved_within_shard() -> None:
    files = _files(8)
    result = _apply_shard_filter(files, {"shard_index": 0, "shard_count": 2})
    assert result == sorted(result, key=lambda f: files.index(f))
