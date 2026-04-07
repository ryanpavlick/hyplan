"""Tests for hyplan.terrain (unit-testable parts, no network)."""

import os
import pytest
import tempfile

from hyplan.terrain import get_cache_root, clear_cache, _COS_TILT_MIN


class TestGetCacheRoot:
    def test_default_path(self):
        root = get_cache_root()
        assert root.endswith("hyplan")
        assert tempfile.gettempdir() in root

    def test_custom_path(self):
        root = get_cache_root(custom_path="/tmp/custom_hyplan_cache")
        assert root == "/tmp/custom_hyplan_cache"

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("HYPLAN_CACHE_ROOT", "/tmp/env_cache")
        root = get_cache_root()
        assert root == "/tmp/env_cache"

    def test_custom_takes_precedence(self, monkeypatch):
        monkeypatch.setenv("HYPLAN_CACHE_ROOT", "/tmp/env_cache")
        root = get_cache_root(custom_path="/tmp/override")
        assert root == "/tmp/override"


class TestClearCache:
    def test_clear_nonexistent(self, monkeypatch):
        """Clearing a non-existent cache under tempdir should not raise."""
        nonexistent = os.path.join(tempfile.gettempdir(), "hyplan_test_nonexistent")
        monkeypatch.setenv("HYPLAN_CACHE_ROOT", nonexistent)
        clear_cache()  # Should not raise since dir doesn't exist

    def test_clear_existing(self, monkeypatch, tmp_path):
        """Should remove the cache directory."""
        cache_dir = tmp_path / "hyplan"
        cache_dir.mkdir()
        (cache_dir / "test.txt").write_text("data")
        monkeypatch.setenv("HYPLAN_CACHE_ROOT", str(cache_dir))
        # clear_cache checks that the path starts with tempdir
        # Since tmp_path is under tempdir, this should work
        # But the safety check requires startswith(tempfile.gettempdir())
        if str(cache_dir).startswith(tempfile.gettempdir()):
            clear_cache()
            assert not cache_dir.exists()

    def test_refuses_unsafe_path(self, monkeypatch):
        """Should refuse to clear a directory outside tempdir."""
        monkeypatch.setenv("HYPLAN_CACHE_ROOT", "/home/user/important_data")
        with pytest.raises(ValueError, match="unsafe"):
            clear_cache()


class TestConstants:
    def test_cos_tilt_min(self):
        assert _COS_TILT_MIN > 0
        assert _COS_TILT_MIN < 0.01
