# (c) 2025 Mario Sieg. <mario.sieg.64@gmail.com>

# Intel
mag_register_cpu_backend("amd64/mag_cpu_amd64_alderlake.c"       "-march=alderlake -mtune=alderlake"             "/arch:AVX2")   # AVX-VNNI (no AVX-512 on most SKUs)
mag_register_cpu_backend("amd64/mag_cpu_amd64_arrowlake.c"       "-march=arrowlake -mtune=arrowlake"             "/arch:AVX2")   # modern client, AVX-VNNI
mag_register_cpu_backend("amd64/mag_cpu_amd64_cannonlake.c"      "-march=cannonlake -mtune=cannonlake"           "/arch:AVX512") # client AVX-512 (IFMA/VBMI)
mag_register_cpu_backend("amd64/mag_cpu_amd64_cascadelake.c"     "-march=cascadelake -mtune=cascadelake"         "/arch:AVX512") # AVX-512 + VNNI
mag_register_cpu_backend("amd64/mag_cpu_amd64_cooperlake.c"      "-march=cooperlake -mtune=cooperlake"           "/arch:AVX512") # AVX-512 + BF16
mag_register_cpu_backend("amd64/mag_cpu_amd64_core2.c"           "-march=core2 -mtune=core2"                     "/arch:SSE2")   # SSSE3-era (MSVC has no /arch:SSE4.x)
mag_register_cpu_backend("amd64/mag_cpu_amd64_haswell.c"         "-march=haswell -mtune=haswell"                 "/arch:AVX2")   # AVX2+FMA
mag_register_cpu_backend("amd64/mag_cpu_amd64_icelake.c"         "-march=icelake-client -mtune=icelake-client"   "/arch:AVX512") # client/server AVX-512 superset
mag_register_cpu_backend("amd64/mag_cpu_amd64_ivybridge.c"       "-march=ivybridge -mtune=ivybridge"             "/arch:AVX")    # AVX + F16C
mag_register_cpu_backend("amd64/mag_cpu_amd64_nehalem.c"         "-march=nehalem -mtune=nehalem"                 "/arch:SSE4.2")  # SSE4.2
mag_register_cpu_backend("amd64/mag_cpu_amd64_sandybridge.c"     "-march=sandybridge -mtune=sandybridge"         "/arch:AVX")    # first AVX
mag_register_cpu_backend("amd64/mag_cpu_amd64_sapphirerapids.c"  "-march=sapphirerapids -mtune=sapphirerapids"   "/arch:AVX512") # AVX512-FP16; AMX via intrinsics
mag_register_cpu_backend("amd64/mag_cpu_amd64_sierraforest.c"    "-march=sierraforest -mtune=sierraforest"       "/arch:AVX2")   # E-core server: AVX2 + AVX-VNNI
mag_register_cpu_backend("amd64/mag_cpu_amd64_skylake_avx512.c"  "-march=skylake-avx512 -mtune=skylake-avx512"   "/arch:AVX512") # base AVX-512 (F+VL+DQ+BW)
mag_register_cpu_backend("amd64/mag_cpu_amd64_tigerlake.c"       "-march=tigerlake -mtune=tigerlake"             "/arch:AVX512") # client AVX-512 (VNNI etc.)

# AMD
mag_register_cpu_backend("amd64/mag_cpu_amd64_zn1.c"             "-march=znver1 -mtune=znver1"                   "/arch:AVX2")   # Zen/Zen+
mag_register_cpu_backend("amd64/mag_cpu_amd64_zn2.c"             "-march=znver2 -mtune=znver2"                   "/arch:AVX2")
mag_register_cpu_backend("amd64/mag_cpu_amd64_zn3.c"             "-march=znver3 -mtune=znver3"                   "/arch:AVX2")
mag_register_cpu_backend("amd64/mag_cpu_amd64_zn4.c"             "-march=znver4 -mtune=znver4"                   "/arch:AVX512") # AVX-512 present on Zen 4 EPYC/Ryzen 7000
mag_register_cpu_backend("amd64/mag_cpu_amd64_zn5.c"             "-march=znver5 -mtune=znver5"                   "/arch:AVX512") # AVX-VNNI + (AVX-512 on many SKUs)

