# (c) 2025 Mario Sieg. <mario.sieg.64@gmail.com>

mag_register_cpu_backend("arm64/mag_cpu_arm64_v82.c" "-march=armv8.2-a+dotprod+fp16" "")
mag_register_cpu_backend("arm64/mag_cpu_arm64_v86.c" "-march=armv8.6-a+bf16+i8mm+fp16+dotprod" "")
if(NOT APPLE) # Skip on Apple (LLVM crash and no HW support)
    mag_register_cpu_backend("arm64/mag_cpu_arm64_v82_sve.c" "-march=armv8.2-a+sve" "")
    mag_register_cpu_backend("arm64/mag_cpu_arm64_v9_sve2.c" "-march=armv9-a+sve2" "")
endif()
