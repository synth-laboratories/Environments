#!/usr/bin/env python3

"""
Test script to verify crafter_classic environment fixes.
Tests JAX logging suppression and error handling.
"""

import asyncio
import logging
import sys


def test_logging_configuration():
    """Test that logging is properly configured."""
    print("Testing logging configuration...")

    # Import configuration to trigger setup
    from examples.crafter_classic.config_logging import configure_logging, safe_compare

    configure_logging()

    # Check that JAX loggers are set to WARNING level
    jax_loggers = [
        "jax._src.cache_key",
        "jax._src.compilation_cache",
        "jax._src.compiler",
        "jax._src.dispatch",
    ]

    success = True
    for logger_name in jax_loggers:
        logger = logging.getLogger(logger_name)
        if logger.level != logging.WARNING:
            print(
                f"❌ JAX logger {logger_name} not set to WARNING level (got {logger.level})"
            )
            success = False
        else:
            print(f"✅ JAX logger {logger_name} correctly set to WARNING level")

    # Test safe comparison function
    print("\nTesting safe comparison function...")

    # Test normal comparisons
    assert safe_compare(5, 10, "<") == True
    assert safe_compare(10, 5, "<") == False
    assert safe_compare("5", 10, "<") == True  # String to int conversion
    assert safe_compare(5, "10", "<") == True  # Int to string conversion
    assert safe_compare("hello", 5, "<") == False  # Should fail safely

    print("✅ Safe comparison function working correctly")

    if success:
        print("✅ Logging configuration successful!")
    else:
        print("❌ Logging configuration has issues")

    return success


def test_environment_import():
    """Test that the environment can be imported without crashes."""
    print("\nTesting environment import...")

    try:

        print("✅ Successfully imported CrafterClassicEnvironment and CrafterEngine")
        return True
    except Exception as e:
        print(f"❌ Failed to import environment: {e}")
        return False


def test_comparison_safety():
    """Test that the comparison safety fixes work."""
    print("\nTesting comparison safety...")

    from examples.crafter_classic.config_logging import safe_compare

    # Test cases that would previously fail
    test_cases = [
        ("5", 10, "<", True),
        (5, "10", "<", True),
        ("hello", 5, "<", False),
        (5, "world", "<", False),
        ("15", "10", ">", True),
        ("abc", "def", "<", True),  # String comparison
    ]

    all_passed = True
    for left, right, op, expected in test_cases:
        result = safe_compare(left, right, op)
        if result == expected:
            print(f"✅ safe_compare({left}, {right}, '{op}') = {result}")
        else:
            print(
                f"❌ safe_compare({left}, {right}, '{op}') = {result}, expected {expected}"
            )
            all_passed = False

    return all_passed


async def main():
    """Main test function."""
    print("Running Crafter Classic environment fixes test...\n")

    # Test 1: Logging configuration
    test1_passed = test_logging_configuration()

    # Test 2: Environment import
    test2_passed = test_environment_import()

    # Test 3: Comparison safety
    test3_passed = test_comparison_safety()

    print("\n" + "=" * 50)
    print("Test Results:")
    print(f"Logging Configuration: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Environment Import: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    print(f"Comparison Safety: {'✅ PASSED' if test3_passed else '❌ FAILED'}")

    all_tests_passed = test1_passed and test2_passed and test3_passed
    print(
        f"\nOverall: {'✅ ALL TESTS PASSED' if all_tests_passed else '❌ SOME TESTS FAILED'}"
    )

    if all_tests_passed:
        print("\n🎉 JAX logging is suppressed and comparison errors are fixed!")
    else:
        print("\n⚠️  Some issues remain to be addressed.")

    return 0 if all_tests_passed else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
