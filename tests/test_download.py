"""Tests for hyplan.download."""

import contextlib
import os

import pytest

from hyplan.download import download_file


class TestDownloadFile:
    def test_skips_existing_file(self, tmp_path):
        """Should not re-download if file exists and replace=False."""
        filepath = str(tmp_path / "existing.txt")
        with open(filepath, "w") as f:
            f.write("original content")

        # Call with a bogus URL — should skip download and not raise
        download_file(filepath, "https://invalid.example.com/bogus", replace=False)

        with open(filepath) as f:
            assert f.read() == "original content"

    def test_creates_directory(self, tmp_path):
        """Should create parent directories if they don't exist."""
        filepath = str(tmp_path / "subdir" / "nested" / "file.txt")
        # Will fail on actual download, but directory creation happens first
        with contextlib.suppress(Exception):  # Expected to fail on network
            download_file(filepath, "https://invalid.example.com/bogus", timeout=1)
        # Parent directory should have been created
        assert os.path.isdir(os.path.dirname(filepath))

    def test_replace_flag(self, tmp_path):
        """With replace=True, download should be attempted even if file exists."""
        filepath = str(tmp_path / "replaceable.txt")
        with open(filepath, "w") as f:
            f.write("old content")

        # This should attempt the download (and fail due to bad URL)
        with pytest.raises(Exception):
            download_file(filepath, "https://invalid.example.com/bogus", replace=True, timeout=1)

    def test_successful_download_writes_file(self, tmp_path, monkeypatch):
        """Mock requests.get to verify the success path: chunked write,
        atomic rename from .tmp, no leftover .tmp file."""
        import requests

        class _FakeResponse:
            def __init__(self):
                self._chunks = [b"hello ", b"world"]

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def raise_for_status(self):
                pass

            def iter_content(self, chunk_size):
                yield from self._chunks

        def _fake_get(url, stream, timeout):
            return _FakeResponse()

        monkeypatch.setattr(requests, "get", _fake_get)

        filepath = str(tmp_path / "out.bin")
        download_file(filepath, "https://example.com/whatever", timeout=5)

        # File written with full content; .tmp cleaned up.
        assert os.path.exists(filepath)
        with open(filepath, "rb") as f:
            assert f.read() == b"hello world"
        assert not os.path.exists(filepath + ".tmp")

    def test_failed_download_cleans_up_tmp(self, tmp_path, monkeypatch):
        """Mock requests.get to raise mid-download; verify the .tmp
        file is removed on failure and the exception propagates."""
        import requests

        class _FakeBadResponse:
            def __init__(self):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def raise_for_status(self):
                pass

            def iter_content(self, chunk_size):
                yield b"partial"
                raise requests.RequestException("simulated network failure")

        def _fake_get(url, stream, timeout):
            return _FakeBadResponse()

        monkeypatch.setattr(requests, "get", _fake_get)

        filepath = str(tmp_path / "incomplete.bin")
        with pytest.raises(requests.RequestException):
            download_file(filepath, "https://example.com/whatever", timeout=5)

        # No partial output, no .tmp leftover.
        assert not os.path.exists(filepath)
        assert not os.path.exists(filepath + ".tmp")
