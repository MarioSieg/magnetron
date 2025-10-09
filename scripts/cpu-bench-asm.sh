#!/usr/bin/env bash

rm perf.data
perf record -F 999 -e cycles:u -g --call-graph dwarf -- bin/release/magnetron_benchmark
perf report