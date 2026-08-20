# (c) 2026 Mario Sieg. <mario.sieg.64@gmail.com>

# SSE4.2 base: Nehalem, Core 2 45nm, Bulldozer, No PCLMUL
mag_register_cpu_backend("amd64/mag_cpu_amd64_v2.c" "-march=x86-64-v2 -mtune=generic" "/arch:SSE2" "v2")

# AVX1: Sandy Bridge, Ivy Bridge, Piledriver, No F16C
mag_register_cpu_backend("amd64/mag_cpu_amd64_v2_avx.c" "-march=x86-64-v2 -mavx -mpclmul -mtune=generic" "/arch:AVX" "v2_avx")

# AVX2 + FMA + F16C + BMI2: Haswell .. Alder/Arrow/Sierra Forest, Zen 1..3
mag_register_cpu_backend("amd64/mag_cpu_amd64_v3.c" "-march=x86-64-v3 -mpclmul -mtune=generic" "/arch:AVX2" "v3")

# AVX-512 F/BW/DQ/VL/CD: Skylake-SP, Cannon/Ice/Tiger Lake
mag_register_cpu_backend("amd64/mag_cpu_amd64_v4.c"  "-march=x86-64-v4 -mpclmul -mtune=generic"  "/arch:AVX512"  "v4")

# AVX-512 + BF16: Cooper Lake, Sapphire Rapids, Zen 4...5
mag_register_cpu_backend("amd64/mag_cpu_amd64_v4_bf16.c" "-march=x86-64-v4 -mavx512bf16 -mpclmul -mtune=generic" "/arch:AVX512" "v4_bf16")
