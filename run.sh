#!/usr/bin/env bash
set -e
cmake --preset default
ln -sf build/compile_commands.json compile_commands.json
cmake --build --preset default
./build/main # target name of the executable
