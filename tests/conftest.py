"""Pytest configuration and fixtures for performax tests."""

import pytest


def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")


@pytest.fixture
def sample_trace_events():
    """Sample trace events for testing."""
    from performax.decorators import PERFORMAX_PREFIX

    return [
        {"ph": "X", "name": f"{PERFORMAX_PREFIX}func1", "dur": 1000},
        {"ph": "X", "name": f"{PERFORMAX_PREFIX}func2", "dur": 2000},
        {"ph": "X", "name": f"{PERFORMAX_PREFIX}func1", "dur": 1500},
    ]


@pytest.fixture
def sample_function_stats():
    """Sample FunctionStats for testing."""
    from performax.result import FunctionStats

    return [
        FunctionStats(name="slow_func", total_duration_ms=500.0, call_count=10),
        FunctionStats(name="fast_func", total_duration_ms=50.0, call_count=100),
        FunctionStats(name="medium_func", total_duration_ms=200.0, call_count=20),
    ]
