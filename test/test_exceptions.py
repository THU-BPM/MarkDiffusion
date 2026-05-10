# Copyright 2025 THU-BPM MarkDiffusion.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Unit tests for custom exceptions in MarkDiffusion.

Tests cover all exception classes defined in exceptions/exceptions.py.
"""

import pytest

from markdiffusion.exceptions.exceptions import (
    LengthMismatchError,
    InvalidTextSourceModeError,
    AlgorithmNameMismatchError,
    InvalidDirectAnalyzerTypeError,
    InvalidReferencedAnalyzerTypeError,
    InvalidAnswerError,
    TypeMismatchException,
    ConfigurationError,
    OpenAIModelConfigurationError,
    DiversityValueError,
    CodeExecutionError,
    InvalidDetectModeError,
    InvalidWatermarkModeError,
)


# ============================================================================
# Tests for LengthMismatchError
# ============================================================================

class TestLengthMismatchError:
    """Tests for LengthMismatchError exception."""

    def test_message_format(self):
        """Test that error message contains expected and actual values."""
        error = LengthMismatchError(expected=10, actual=5)
        message = str(error)
        assert "Expected length: 10" in message
        assert "but got 5" in message

    def test_raises_correctly(self):
        """Test that exception can be raised and caught."""
        with pytest.raises(LengthMismatchError):
            raise LengthMismatchError(expected=100, actual=50)

    def test_inheritance(self):
        """Test that LengthMismatchError inherits from Exception."""
        error = LengthMismatchError(10, 5)
        assert isinstance(error, Exception)

    def test_different_values(self):
        """Test with different expected and actual values."""
        test_cases = [(0, 1), (100, 0), (1000, 999), (1, 1000000)]
        for expected, actual in test_cases:
            error = LengthMismatchError(expected, actual)
            assert str(expected) in str(error)
            assert str(actual) in str(error)


# ============================================================================
# Tests for InvalidTextSourceModeError
# ============================================================================

class TestInvalidTextSourceModeError:
    """Tests for InvalidTextSourceModeError exception."""

    def test_message_format(self):
        """Test that error message contains the invalid mode."""
        error = InvalidTextSourceModeError("invalid_mode")
        message = str(error)
        assert "'invalid_mode' is not a valid text source mode" in message
        assert "natural" in message
        assert "generated" in message

    def test_inheritance(self):
        """Test that InvalidTextSourceModeError inherits from ValueError."""
        error = InvalidTextSourceModeError("test")
        assert isinstance(error, ValueError)

    def test_raises_correctly(self):
        """Test that exception can be raised and caught."""
        with pytest.raises(InvalidTextSourceModeError):
            raise InvalidTextSourceModeError("bad_mode")

    def test_various_invalid_modes(self):
        """Test with various invalid mode names."""
        invalid_modes = ["invalid", "", "Natural", "GENERATED", "random", "123"]
        for mode in invalid_modes:
            error = InvalidTextSourceModeError(mode)
            assert f"'{mode}'" in str(error)


# ============================================================================
# Tests for AlgorithmNameMismatchError
# ============================================================================

class TestAlgorithmNameMismatchError:
    """Tests for AlgorithmNameMismatchError exception."""

    def test_message_format(self):
        """Test that error message contains expected and actual algorithm names."""
        error = AlgorithmNameMismatchError(expected="TR", actual="GS")
        message = str(error)
        assert "TR" in message
        assert "GS" in message
        assert "does not match" in message

    def test_inheritance(self):
        """Test that AlgorithmNameMismatchError inherits from ValueError."""
        error = AlgorithmNameMismatchError("A", "B")
        assert isinstance(error, ValueError)

    def test_raises_correctly(self):
        """Test that exception can be raised and caught."""
        with pytest.raises(AlgorithmNameMismatchError):
            raise AlgorithmNameMismatchError(expected="Expected", actual="Actual")

    def test_various_algorithm_names(self):
        """Test with various algorithm name combinations."""
        test_cases = [
            ("TR", "GS"),
            ("VideoShield", "VideoMark"),
            ("PRC", "ROBIN"),
            ("SFW", "SEAL"),
        ]
        for expected, actual in test_cases:
            error = AlgorithmNameMismatchError(expected, actual)
            assert expected in str(error)
            assert actual in str(error)


# ============================================================================
# Tests for InvalidDirectAnalyzerTypeError
# ============================================================================

class TestInvalidDirectAnalyzerTypeError:
    """Tests for InvalidDirectAnalyzerTypeError exception."""

    def test_default_message(self):
        """Test default error message."""
        error = InvalidDirectAnalyzerTypeError()
        assert "DirectTextQualityAnalyzer" in str(error)

    def test_custom_message(self):
        """Test custom error message."""
        custom_msg = "Custom analyzer error message"
        error = InvalidDirectAnalyzerTypeError(custom_msg)
        assert str(error) == custom_msg

    def test_inheritance(self):
        """Test that InvalidDirectAnalyzerTypeError inherits from Exception."""
        error = InvalidDirectAnalyzerTypeError()
        assert isinstance(error, Exception)

    def test_raises_correctly(self):
        """Test that exception can be raised and caught."""
        with pytest.raises(InvalidDirectAnalyzerTypeError):
            raise InvalidDirectAnalyzerTypeError()


# ============================================================================
# Tests for InvalidReferencedAnalyzerTypeError
# ============================================================================

class TestInvalidReferencedAnalyzerTypeError:
    """Tests for InvalidReferencedAnalyzerTypeError exception."""

    def test_default_message(self):
        """Test default error message."""
        error = InvalidReferencedAnalyzerTypeError()
        assert "ReferencedTextQualityAnalyzer" in str(error)

    def test_custom_message(self):
        """Test custom error message."""
        custom_msg = "Custom referenced analyzer error"
        error = InvalidReferencedAnalyzerTypeError(custom_msg)
        assert str(error) == custom_msg

    def test_inheritance(self):
        """Test that InvalidReferencedAnalyzerTypeError inherits from Exception."""
        error = InvalidReferencedAnalyzerTypeError()
        assert isinstance(error, Exception)

    def test_raises_correctly(self):
        """Test that exception can be raised and caught."""
        with pytest.raises(InvalidReferencedAnalyzerTypeError):
            raise InvalidReferencedAnalyzerTypeError()


# ============================================================================
# Tests for InvalidAnswerError
# ============================================================================

class TestInvalidAnswerError:
    """Tests for InvalidAnswerError exception."""

    def test_message_format(self):
        """Test that error message contains the invalid answer."""
        error = InvalidAnswerError("bad_answer")
        assert "Invalid answer: bad_answer" in str(error)

    def test_inheritance(self):
        """Test that InvalidAnswerError inherits from ValueError."""
        error = InvalidAnswerError("test")
        assert isinstance(error, ValueError)

    def test_raises_correctly(self):
        """Test that exception can be raised and caught."""
        with pytest.raises(InvalidAnswerError):
            raise InvalidAnswerError("invalid")

    def test_various_answer_types(self):
        """Test with various answer types."""
        answers = ["string_answer", 123, None, "", ["list"], {"dict": "value"}]
        for answer in answers:
            error = InvalidAnswerError(answer)
            assert "Invalid answer" in str(error)


# ============================================================================
# Tests for TypeMismatchException
# ============================================================================

class TestTypeMismatchException:
    """Tests for TypeMismatchException exception."""

    def test_message_with_types(self):
        """Test error message with expected and found types."""
        error = TypeMismatchException(expected_type=int, found_type=str)
        message = str(error)
        assert "int" in message
        assert "str" in message

    def test_custom_message(self):
        """Test custom error message overrides default."""
        custom_msg = "Custom type mismatch message"
        error = TypeMismatchException(int, str, custom_msg)
        assert str(error) == custom_msg

    def test_attributes_stored(self):
        """Test that expected_type and found_type are stored."""
        error = TypeMismatchException(expected_type=list, found_type=dict)
        assert error.expected_type == list
        assert error.found_type == dict

    def test_inheritance(self):
        """Test that TypeMismatchException inherits from Exception."""
        error = TypeMismatchException(int, str)
        assert isinstance(error, Exception)

    def test_raises_correctly(self):
        """Test that exception can be raised and caught."""
        with pytest.raises(TypeMismatchException):
            raise TypeMismatchException(int, str)

    def test_various_type_combinations(self):
        """Test with various type combinations."""
        type_pairs = [
            (int, str),
            (list, tuple),
            (dict, list),
            (float, int),
            (str, bytes),
        ]
        for expected, found in type_pairs:
            error = TypeMismatchException(expected, found)
            assert expected.__name__ in str(error)
            assert found.__name__ in str(error)


# ============================================================================
# Tests for ConfigurationError
# ============================================================================

class TestConfigurationError:
    """Tests for ConfigurationError exception."""

    def test_message_stored(self):
        """Test that message is stored correctly."""
        error = ConfigurationError("Test configuration error")
        assert error.message == "Test configuration error"
        assert str(error) == "Test configuration error"

    def test_inheritance(self):
        """Test that ConfigurationError inherits from Exception."""
        error = ConfigurationError("test")
        assert isinstance(error, Exception)

    def test_raises_correctly(self):
        """Test that exception can be raised and caught."""
        with pytest.raises(ConfigurationError):
            raise ConfigurationError("Config error")

    def test_various_messages(self):
        """Test with various error messages."""
        messages = [
            "Missing required field",
            "Invalid value for parameter",
            "Configuration file not found",
            "",
        ]
        for msg in messages:
            error = ConfigurationError(msg)
            assert error.message == msg


# ============================================================================
# Tests for OpenAIModelConfigurationError
# ============================================================================

class TestOpenAIModelConfigurationError:
    """Tests for OpenAIModelConfigurationError exception."""

    def test_message_format(self):
        """Test error message format."""
        error = OpenAIModelConfigurationError("Invalid API key")
        assert str(error) == "Invalid API key"

    def test_inheritance(self):
        """Test that OpenAIModelConfigurationError inherits from Exception."""
        error = OpenAIModelConfigurationError("test")
        assert isinstance(error, Exception)

    def test_raises_correctly(self):
        """Test that exception can be raised and caught."""
        with pytest.raises(OpenAIModelConfigurationError):
            raise OpenAIModelConfigurationError("API configuration error")


# ============================================================================
# Tests for DiversityValueError
# ============================================================================

class TestDiversityValueError:
    """Tests for DiversityValueError exception."""

    def test_message_format(self):
        """Test that error message contains diversity type and valid values."""
        error = DiversityValueError("lexical")
        message = str(error)
        assert "lexical" in message
        assert "0, 20, 40, 60, 80, 100" in message

    def test_inheritance(self):
        """Test that DiversityValueError inherits from Exception."""
        error = DiversityValueError("test")
        assert isinstance(error, Exception)

    def test_raises_correctly(self):
        """Test that exception can be raised and caught."""
        with pytest.raises(DiversityValueError):
            raise DiversityValueError("semantic")

    def test_various_diversity_types(self):
        """Test with various diversity type names."""
        diversity_types = ["lexical", "semantic", "syntactic", "custom"]
        for dtype in diversity_types:
            error = DiversityValueError(dtype)
            assert dtype in str(error)


# ============================================================================
# Tests for CodeExecutionError
# ============================================================================

class TestCodeExecutionError:
    """Tests for CodeExecutionError exception."""

    def test_default_message(self):
        """Test default error message."""
        error = CodeExecutionError()
        assert error.message == "Error during code execution"
        assert "Error during code execution" in str(error)

    def test_custom_message(self):
        """Test custom error message."""
        custom_msg = "Specific code execution failure"
        error = CodeExecutionError(custom_msg)
        assert error.message == custom_msg
        assert str(error) == custom_msg

    def test_inheritance(self):
        """Test that CodeExecutionError inherits from Exception."""
        error = CodeExecutionError()
        assert isinstance(error, Exception)

    def test_raises_correctly(self):
        """Test that exception can be raised and caught."""
        with pytest.raises(CodeExecutionError):
            raise CodeExecutionError()


# ============================================================================
# Tests for InvalidDetectModeError
# ============================================================================

class TestInvalidDetectModeError:
    """Tests for InvalidDetectModeError exception."""

    def test_mode_stored(self):
        """Test that mode is stored correctly."""
        error = InvalidDetectModeError("bad_mode")
        assert error.mode == "bad_mode"

    def test_default_message(self):
        """Test default error message format."""
        error = InvalidDetectModeError("test_mode")
        assert error.message == "Invalid detect mode configuration"
        assert "test_mode" in str(error)

    def test_custom_message(self):
        """Test custom error message."""
        error = InvalidDetectModeError("mode", "Custom detect error")
        assert error.message == "Custom detect error"
        assert "mode" in str(error)

    def test_inheritance(self):
        """Test that InvalidDetectModeError inherits from Exception."""
        error = InvalidDetectModeError("test")
        assert isinstance(error, Exception)

    def test_raises_correctly(self):
        """Test that exception can be raised and caught."""
        with pytest.raises(InvalidDetectModeError):
            raise InvalidDetectModeError("invalid")


# ============================================================================
# Tests for InvalidWatermarkModeError
# ============================================================================

class TestInvalidWatermarkModeError:
    """Tests for InvalidWatermarkModeError exception."""

    def test_mode_stored(self):
        """Test that mode is stored correctly."""
        error = InvalidWatermarkModeError("bad_mode")
        assert error.mode == "bad_mode"

    def test_default_message(self):
        """Test default error message format."""
        error = InvalidWatermarkModeError("test_mode")
        assert error.message == "Invalid watermark mode configuration"
        assert "test_mode" in str(error)

    def test_custom_message(self):
        """Test custom error message."""
        error = InvalidWatermarkModeError("mode", "Custom watermark error")
        assert error.message == "Custom watermark error"
        assert "mode" in str(error)

    def test_inheritance(self):
        """Test that InvalidWatermarkModeError inherits from Exception."""
        error = InvalidWatermarkModeError("test")
        assert isinstance(error, Exception)

    def test_raises_correctly(self):
        """Test that exception can be raised and caught."""
        with pytest.raises(InvalidWatermarkModeError):
            raise InvalidWatermarkModeError("invalid")


# ============================================================================
# Integration Tests - Exception Handling Patterns
# ============================================================================

class TestExceptionHandlingPatterns:
    """Test common exception handling patterns."""

    def test_catch_value_errors(self):
        """Test that ValueError subclasses can be caught as ValueError."""
        value_error_exceptions = [
            InvalidTextSourceModeError("test"),
            AlgorithmNameMismatchError("A", "B"),
            InvalidAnswerError("test"),
        ]
        for exc in value_error_exceptions:
            with pytest.raises(ValueError):
                raise exc

    def test_exception_chaining(self):
        """Test exception chaining works correctly."""
        try:
            try:
                raise ValueError("Original error")
            except ValueError as e:
                raise ConfigurationError("Config failed") from e
        except ConfigurationError as e:
            assert e.__cause__ is not None
            assert isinstance(e.__cause__, ValueError)

    def test_exception_in_context_manager(self):
        """Test exceptions work in context managers."""
        class TestContext:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                if exc_type is LengthMismatchError:
                    return True  # Suppress the exception
                return False

        with TestContext():
            raise LengthMismatchError(10, 5)  # Should be suppressed

        with pytest.raises(ConfigurationError):
            with TestContext():
                raise ConfigurationError("Not suppressed")
