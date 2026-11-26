#!/bin/bash

echo "🧪 Depth Surge 3D - Unit Test Runner"
echo "====================================="
echo ""

# Check if virtual environment exists and activate it
if [ -d ".venv" ]; then
    echo "🔧 Activating virtual environment (.venv)..."
    source .venv/bin/activate
elif [ -d "venv" ]; then
    echo "🔧 Activating virtual environment (venv)..."
    source venv/bin/activate
else
    echo "❌ Error: Virtual environment not found."
    echo "   Follow docs/INSTALLATION.md and include the dev group."
    exit 1
fi

# Check if pytest is installed
if ! python -c "import pytest" 2>/dev/null; then
    echo "❌ Error: pytest not installed in virtual environment."
    echo "   Run: pip install pytest pytest-cov"
    exit 1
fi

echo "✅ Virtual environment activated"
echo ""

# Check if pytest-cov is available for coverage
HAS_COV=false
if python -c "import pytest_cov" 2>/dev/null; then
    HAS_COV=true
fi

# Parse command line arguments
PYTEST_ARGS="-v tests/unit"
SHOW_COVERAGE=true  # Show coverage by default

for arg in "$@"; do
    case $arg in
        --coverage|-c)
            SHOW_COVERAGE=true
            shift
            ;;
        --verbose|-v)
            PYTEST_ARGS="$PYTEST_ARGS -vv"
            shift
            ;;
        --help|-h)
            echo "Usage: ./scripts/run-unit-tests.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -c, --coverage    Show coverage report (requires pytest-cov)"
            echo "  -v, --verbose     Verbose output"
            echo "  -h, --help        Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./scripts/run-unit-tests.sh              # Run unit tests"
            echo "  ./scripts/run-unit-tests.sh --coverage   # Run with coverage"
            echo ""
            exit 0
            ;;
        *)
            # Pass through any other arguments to pytest
            PYTEST_ARGS="$PYTEST_ARGS $arg"
            shift
            ;;
    esac
done

# Add coverage args if requested and available
if [ "$SHOW_COVERAGE" = true ]; then
    if [ "$HAS_COV" = true ]; then
        echo "📊 Coverage reporting enabled"
        PYTEST_ARGS="$PYTEST_ARGS --cov=src/depth_surge_3d --cov-report=term-missing --cov-report=html"
    else
        echo "⚠️  Warning: pytest-cov not installed. Running without coverage."
        echo "   Install with: pip install pytest-cov"
        echo ""
    fi
fi

# Run the tests
echo "🚀 Running unit tests..."
echo "   Command: pytest $PYTEST_ARGS"
echo ""

pytest $PYTEST_ARGS

# Capture exit code
EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ All unit tests passed!"

    if [ "$SHOW_COVERAGE" = true ] && [ "$HAS_COV" = true ]; then
        echo ""
        echo "📊 Coverage report generated:"
        echo "   Terminal: See above"
        echo "   HTML: htmlcov/index.html"
        echo ""
        echo "💡 Open coverage report:"
        echo "   xdg-open htmlcov/index.html  (Linux)"
        echo "   open htmlcov/index.html       (macOS)"
    fi
else
    echo "❌ Some unit tests failed (exit code: $EXIT_CODE)"
    echo ""
    echo "💡 Tips:"
    echo "   - Review the test output above for details"
    echo "   - Run with --verbose for more information"
    echo "   - Check that all dependencies are installed"
fi

echo ""
exit $EXIT_CODE
