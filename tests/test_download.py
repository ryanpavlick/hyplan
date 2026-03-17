"""Tests for hyplan.download."""

import os
import pytest
import tempfile

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
        try:
            download_file(filepath, "https://invalid.example.com/bogus", timeout=1)
        except Exception:
            pass  # Expected to fail on network
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
