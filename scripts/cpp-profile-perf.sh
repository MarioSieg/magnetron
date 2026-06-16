#!/usr/bin/env bash

rm perf.data
perf record -F 999 -e cycles:u -g --call-graph dwarf -- bin/release/magnetron_benchmark
perf report --stdio > perf_report.txt
perf annotate --stdio --symbol=mag_gemv_vec_mat_fp8w_scaled_kernel_rhs_transposed_contig_mag_bfloat16_t > annotate.txt
