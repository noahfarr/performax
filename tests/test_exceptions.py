"""Tests for performax exceptions."""

import pytest

from performax.exceptions import ProfilingError


class TestProfilingError:
    """Tests for the ProfilingError exception."""

    def test_inherits_from_exception(self):
        """Test that ProfilingError inherits from Exception."""
        assert issubclass(ProfilingError, Exception)

    def test_can_be_raised(self):
        """Test that ProfilingError can be raised and caught."""
        with pytest.raises(ProfilingError):
            raise ProfilingError("test error")

    def test_message_preserved(self):
        """Test that error message is preserved."""
        try:
            raise ProfilingError("specific error message")
        except ProfilingError as e:
            assert str(e) == "specific error message"

    def test_can_catch_as_exception(self):
        """Test that ProfilingError can be caught as Exception."""
        with pytest.raises(Exception):
            raise ProfilingError("test")

    def test_empty_message(self):
        """Test ProfilingError with empty message."""
        error = ProfilingError("")
        assert str(error) == ""

    def test_with_cause(self):
        """Test ProfilingError with __cause__."""
        original = ValueError("original error")
        try:
            try:
                raise original
            except ValueError as e:
                raise ProfilingError("wrapped error") from e
        except ProfilingError as e:
            assert e.__cause__ is original
