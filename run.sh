#!/usr/bin/env bash
set -e
cmake --preset release
ln -sf build/release/compile_commands.json compile_commands.json
cmake --build --preset release -j
./build/release/main
