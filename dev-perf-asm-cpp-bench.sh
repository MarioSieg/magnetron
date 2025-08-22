#!/usr/bin/env bash
set -euo pipefail

rm perf.data
perf record -F 999 -e cycles:u -g --call-graph dwarf -- bin/release/magnetron_benchmark
perf report