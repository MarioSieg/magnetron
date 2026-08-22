

# ARMv8.2 baseline: Cortex-A75/A76, Neoverse N1, Apple M1 class
mag_register_cpu_backend("arm64/mag_cpu_arm64_v82.c" "-march=armv8.2-a+dotprod+fp16" ""  "arm_v82")

# + BF16 and I8MM: Neoverse V1/N2/V2, Cortex-X2+, Apple M2+
mag_register_cpu_backend("arm64/mag_cpu_arm64_v86.c" "-march=armv8.6-a+bf16+i8mm+fp16+dotprod" ""  "arm_v86")

# + AES/PMULL for folded CRC32C.
mag_register_cpu_backend("arm64/mag_cpu_arm64_v86_crypto.c" "-march=armv8.6-a+bf16+i8mm+fp16+dotprod+crypto" ""  "arm_v86_crypto")

# Skip SVE impls on Apple (LLVM crash and no hardware support)
if (NOT APPLE)
    # + SVE: Neoverse V1 (Graviton 3), Neoverse N2
    mag_register_cpu_backend("arm64/mag_cpu_arm64_v86_sve.c" "-march=armv8.6-a+sve+bf16+i8mm+fp16+dotprod+crypto" ""  "arm_v86_sve")

    # + SVE2: Neoverse V2 (Graviton 4), Cortex-X3+, Snapdragon X
    mag_register_cpu_backend("arm64/mag_cpu_arm64_v9_sve2.c" "-march=armv9-a+sve2+bf16+i8mm+fp16+dotprod+crypto" ""  "arm_v9_sve2")

    # SVE without dotprod/bf16/i8mm: A64FX
    mag_register_cpu_backend("arm64/mag_cpu_arm64_v82_sve.c" "-march=armv8.2-a+sve" ""  "arm_v82_sve")
endif()
