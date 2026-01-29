"""Tests for the track decorator."""

import pytest

from performax.decorators import PERFORMAX_PREFIX, track


class TestTrackDecorator:
    """Tests for the @track decorator."""

    def test_track_without_arguments(self):
        """Test @track decorator without arguments uses function name."""

        @track
        def my_function():
            return 42

        assert my_function() == 42
        assert my_function.__name__ == "my_function"

    def test_track_with_custom_name(self):
        """Test @track(name='custom') decorator with custom name."""

        @track(name="custom_name")
        def my_function():
            return 42

        assert my_function() == 42
        assert my_function.__name__ == "my_function"

    def test_track_preserves_return_value(self):
        """Test that decorated function returns correct value."""

        @track
        def add(a, b):
            return a + b

        assert add(2, 3) == 5

    def test_track_preserves_args_and_kwargs(self):
        """Test that decorated function receives correct arguments."""

        @track
        def func_with_args(a, b, c=10, d=20):
            return a + b + c + d

        assert func_with_args(1, 2) == 33
        assert func_with_args(1, 2, c=100) == 123
        assert func_with_args(1, 2, c=100, d=200) == 303

    def test_track_preserves_docstring(self):
        """Test that decorated function preserves docstring."""

        @track
        def documented_function():
            """This is a docstring."""
            pass

        assert documented_function.__doc__ == """This is a docstring."""

    def test_track_with_exception(self):
        """Test that exceptions propagate through the decorator."""

        @track
        def raising_function():
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            raising_function()

    def test_track_on_generator(self):
        """Test @track on a function that returns a generator."""

        @track
        def gen_function():
            yield 1
            yield 2
            yield 3

        result = list(gen_function())
        assert result == [1, 2, 3]

    def test_track_on_lambda_equivalent(self):
        """Test @track on simple functions similar to lambdas."""

        @track
        def identity(x):
            return x

        assert identity(42) == 42
        assert identity("hello") == "hello"
        assert identity([1, 2, 3]) == [1, 2, 3]

    def test_track_nested_calls(self):
        """Test multiple tracked functions calling each other."""

        @track
        def inner():
            return 10

        @track
        def outer():
            return inner() * 2

        assert outer() == 20

    def test_track_with_class_method(self):
        """Test @track on class methods."""

        class MyClass:
            @track
            def method(self, x):
                return x * 2

        obj = MyClass()
        assert obj.method(5) == 10

    def test_track_with_static_method(self):
        """Test @track on static methods."""

        class MyClass:
            @staticmethod
            @track
            def static_method(x):
                return x * 3

        assert MyClass.static_method(5) == 15

    def test_performax_prefix_constant(self):
        """Test that PERFORMAX_PREFIX is defined correctly."""
        assert PERFORMAX_PREFIX == "performax/"
        assert PERFORMAX_PREFIX.endswith("/")
