#!/bin/bash
# Run clang-format (in place) and clang-tidy over the SOFIE sources.
set -e

# Always operate from the repository root, wherever the script is invoked from.
cd "$(dirname "${BASH_SOURCE[0]}")/.."

SRC_DIRS="./core ./parsers"
TEST_DIR="./tests"

echo "📝 Discovering source/header files..."

FILES=$(find $SRC_DIRS "$TEST_DIR" \
    -path "$TEST_DIR/models/onnx/references" -prune -o \
    -type f \( \
        -name '*.cpp' -o -name '*.cc' -o -name '*.cxx' -o \
        -name '*.h' -o -name '*.hpp' -o -name '*.hxx' -o -name '*.hh' \
    \) -print)

if [ -z "$FILES" ]; then
    echo "⚠️ No files found to process."
    exit 0
fi

echo "🎯 Files to check:"
echo "$FILES"

echo "🎨 Running clang-format..."
for file in $FILES; do
    echo "Formatting $file"
    clang-format -i "$file"
done

echo "🔍 Running clang-tidy..."
for file in $FILES; do
    echo "Linting $file"
    clang-tidy "$file" --extra-arg=-std=c++20 -- -I./core/inc -I./parsers/inc || true
done

echo "✅ Formatting and linting complete."
