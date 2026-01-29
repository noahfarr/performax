"""Tests for the profile function."""

import threading
import time

import pytest

from performax import ProfileResult, ProfilingError, profile, track


class TestProfileBasic:
    """Basic tests for the profile function."""

    def test_profile_returns_tuple(self):
        """Test that profile returns a tuple of (result, ProfileResult)."""

        def simple_func():
            return 42

        result, stats = profile(simple_func)
        assert result == 42
        assert isinstance(stats, ProfileResult)

    def test_profile_with_tracked_function(self):
        """Test profiling a tracked function."""

        @track
        def tracked_func():
            return "hello"

        def main():
            return tracked_func()

        result, stats = profile(main)
        assert result == "hello"
        assert isinstance(stats, ProfileResult)

    def test_profile_passes_args(self):
        """Test that profile passes positional arguments."""

        def func_with_args(a, b):
            return a + b

        result, _ = profile(func_with_args, 3, 4)
        assert result == 7

    def test_profile_passes_kwargs(self):
        """Test that profile passes keyword arguments."""

        def func_with_kwargs(a, b=10):
            return a * b

        result, _ = profile(func_with_kwargs, 5, b=3)
        assert result == 15

    def test_profile_with_no_tracked_functions(self):
        """Test profiling when no functions are tracked."""

        def untracked():
            return 100

        result, stats = profile(untracked)
        assert result == 100
        assert len(stats.stats) == 0


class TestProfileWithJax:
    """Tests for profile with actual JAX operations."""

    def test_profile_with_jax_array(self):
        """Test profiling returns JAX array correctly."""
        jnp = pytest.importorskip("jax.numpy")

        @track
        def create_array():
            return jnp.ones((10, 10))

        result, stats = profile(create_array)
        assert result.shape == (10, 10)

    def test_profile_with_jax_computation(self):
        """Test profiling JAX computation."""
        jnp = pytest.importorskip("jax.numpy")

        @track
        def matmul():
            a = jnp.ones((100, 100))
            b = jnp.ones((100, 100))
            return jnp.dot(a, b)

        result, stats = profile(matmul)
        assert result.shape == (100, 100)

    def test_profile_captures_tracked_function_stats(self):
        """Test that tracked function appears in stats."""
        jnp = pytest.importorskip("jax.numpy")

        @track(name="test_computation")
        def computation():
            a = jnp.ones((50, 50))
            return jnp.sum(a)

        _, stats = profile(computation)

        # The function should appear in the stats
        names = [s.name for s in stats.stats]
        assert "test_computation" in names

    def test_profile_with_tuple_return(self):
        """Test profiling function that returns tuple of arrays."""
        jnp = pytest.importorskip("jax.numpy")

        @track
        def multi_return():
            return jnp.ones((5,)), jnp.zeros((5,))

        result, stats = profile(multi_return)
        assert len(result) == 2
        assert result[0].shape == (5,)
        assert result[1].shape == (5,)

    def test_profile_nested_tracked_functions(self):
        """Test profiling with nested tracked functions."""
        jnp = pytest.importorskip("jax.numpy")

        @track(name="inner")
        def inner(x):
            return x * 2

        @track(name="outer")
        def outer():
            x = jnp.ones((10,))
            return inner(x)

        result, stats = profile(outer)
        assert result.shape == (10,)

        names = [s.name for s in stats.stats]
        assert "inner" in names
        assert "outer" in names


class TestProfileErrorHandling:
    """Tests for error handling in profile function."""

    def test_profile_wraps_exception(self):
        """Test that exceptions are wrapped in ProfilingError."""

        def failing_func():
            raise ValueError("intentional error")

        with pytest.raises(ProfilingError, match="Profiling failed"):
            profile(failing_func)

    def test_profile_preserves_original_exception(self):
        """Test that original exception is preserved as __cause__."""

        def failing_func():
            raise TypeError("type error")

        try:
            profile(failing_func)
        except ProfilingError as e:
            assert isinstance(e.__cause__, TypeError)
            assert "type error" in str(e.__cause__)


class TestProfileThreadSafety:
    """Tests for thread safety of the profile function."""

    def test_concurrent_profile_raises_error(self):
        """Test that concurrent profiling raises ProfilingError."""
        jnp = pytest.importorskip("jax.numpy")

        started = threading.Event()
        error_raised = threading.Event()
        errors = []

        def slow_profile():
            @track
            def slow_func():
                started.set()
                time.sleep(0.5)
                return jnp.ones((10,))

            try:
                profile(slow_func)
            except ProfilingError as e:
                errors.append(e)
                error_raised.set()

        def concurrent_profile():
            started.wait(timeout=2.0)
            time.sleep(0.1)  # Ensure first profile has started

            def simple_func():
                return 1

            try:
                profile(simple_func)
            except ProfilingError as e:
                errors.append(e)
                error_raised.set()

        t1 = threading.Thread(target=slow_profile)
        t2 = threading.Thread(target=concurrent_profile)

        t1.start()
        t2.start()

        t1.join(timeout=3.0)
        t2.join(timeout=3.0)

        # One of them should have raised an error
        assert len(errors) == 1
        assert "already in progress" in str(errors[0])

    def test_lock_released_after_exception(self):
        """Test that lock is released even when exception occurs."""

        def failing_func():
            raise RuntimeError("test failure")

        # First call should fail
        with pytest.raises(ProfilingError):
            profile(failing_func)

        # Second call should work (lock was released)
        def working_func():
            return 42

        result, _ = profile(working_func)
        assert result == 42

    def test_sequential_profiles_work(self):
        """Test that sequential profile calls work correctly."""

        def func1():
            return 1

        def func2():
            return 2

        def func3():
            return 3

        r1, _ = profile(func1)
        r2, _ = profile(func2)
        r3, _ = profile(func3)

        assert r1 == 1
        assert r2 == 2
        assert r3 == 3


class TestProfileIntegration:
    """Integration tests for the full profiling workflow."""

    def test_full_workflow(self):
        """Test complete profiling workflow with multiple functions."""
        jnp = pytest.importorskip("jax.numpy")

        @track
        def add(a, b):
            return a + b

        @track
        def multiply(a, b):
            return a * b

        @track(name="computation")
        def compute():
            x = jnp.ones((100, 100))
            y = jnp.ones((100, 100))
            z = add(x, y)
            return multiply(z, z)

        result, stats = profile(compute)

        # Check result
        assert result.shape == (100, 100)

        # Check stats
        assert len(stats.stats) >= 1

        # Check output formats work
        dict_output = stats.to_dict()
        assert isinstance(dict_output, list)

        str_output = str(stats)
        assert "Function" in str_output

    def test_profile_result_accuracy(self):
        """Test that profile captures timing for tracked functions."""
        jnp = pytest.importorskip("jax.numpy")

        call_count = 0

        @track(name="counted_func")
        def counted_func():
            nonlocal call_count
            call_count += 1
            return jnp.ones((10,))

        def main():
            for _ in range(5):
                counted_func()
            return counted_func()

        result, stats = profile(main)

        # Function was called 6 times
        assert call_count == 6

        # Find the counted_func in stats
        counted_stats = [s for s in stats.stats if s.name == "counted_func"]
        if counted_stats:
            assert counted_stats[0].call_count == 6
