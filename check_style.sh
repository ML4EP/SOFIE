#!/bin/bash
set -e

# Directories (SOFIE's own layout: source lives under core/parsers/utils,
# tests under test/ - unlike sofieBLAS this script was originally written for,
# SOFIE has no top-level include/ or tests/ directory)
SRC_DIRS="./core ./parsers ./utils"
TEST_DIR="./test"

echo "📝 Discovering source/header files..."

FILES=$(find $SRC_DIRS "$TEST_DIR" \
    -path "$TEST_DIR/build" -prune -o \
    -path "$TEST_DIR/input_models/references" -prune -o \
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
    clang-tidy "$file" --extra-arg=-std=c++20 -- -I./core/inc -I./parsers/inc -I./utils || true
done

echo "✅ Formatting and linting complete."
