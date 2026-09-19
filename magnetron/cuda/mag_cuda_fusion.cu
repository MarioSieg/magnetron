#include "mag_cuda_fusion.cuh"

#include <nvrtc.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace mag {
  namespace {
    constexpr int FUSED_BLOCK_SIZE = 256;

    /*
    ** CUDA expression for each instruction.
    **
    ** $0..$2 are the operands, already in named locals, so an operand used twice is evaluated once.
    ** Each is the arithmetic the eager kernel performs, written the same way round. Reassociating it
    ** would be a different expression and a different answer.
    */
    const char *fused_form(uint8_t op) noexcept {
      switch (op) {
        case MAG_OP_ADD:   return "($0 + $1)";
        case MAG_OP_SUB:   return "($0 - $1)";
        case MAG_OP_MUL:   return "($0 * $1)";
        case MAG_OP_DIV:   return "($0 / $1)";
        case MAG_OP_MIN:   return "fminf($0, $1)";
        case MAG_OP_MAX:   return "fmaxf($0, $1)";
        case MAG_OP_NEG:   return "(-$0)";
        case MAG_OP_ABS:   return "fabsf($0)";
        case MAG_OP_SGN:   return "(float)(($0 > 0.0f) - ($0 < 0.0f))";
        case MAG_OP_SQR:   return "($0 * $0)";
        case MAG_OP_SQRT:  return "sqrtf($0)";
        case MAG_OP_FLOOR: return "floorf($0)";
        case MAG_OP_CEIL:  return "ceilf($0)";
        case MAG_OP_ROUND: return "roundf($0)";
        case MAG_OP_TRUNC: return "truncf($0)";
        case MAG_OP_STEP:  return "($0 > 0.0f ? 1.0f : 0.0f)";
        case MAG_OP_RELU:  return "fmaxf($0, 0.0f)";
        case MAG_OP_CLAMP: return "fminf(fmaxf($0, $1), $2)";
        default: return nullptr;
      }
    }

    void operand_name(const mag_fuse_operand_t &o, char *buf, size_t cap) noexcept {
      switch (o.kind) {
        case MAG_FUSE_REG: snprintf(buf, cap, "r%u", o.idx); break;
        case MAG_FUSE_IMM: snprintf(buf, cap, "s%u", o.idx); break;
        case MAG_FUSE_SCL: snprintf(buf, cap, "mag_ld(b%u[0])", o.idx); break;
        default:           snprintf(buf, cap, "mag_ld(b%u[i])", o.idx); break;
      }
    }

    [[nodiscard]] bool fused_supported(const mag_fuse_graph_t &g) noexcept {
      /* bfloat16 is absent for the reason the CPU backend gives: eager bfloat16 is not yet a
         function of its input alone, so there is nothing definite for a chain to reproduce. */
      if (g.dtype != MAG_DTYPE_FLOAT32 && g.dtype != MAG_DTYPE_FLOAT16) return false;
      if (!g.num_ins || !g.num_stores) return false;
      for (uint32_t i = 0; i < g.num_ins; ++i)
        if (g.ins[i].op != MAG_FUSE_OP_LOAD && !fused_form(g.ins[i].op)) return false;
      return true;
    }

    /*
    ** The chain as one kernel.
    **
    ** A grid-stride loop rather than one thread per element, so the launch geometry does not have to
    ** track the tensor: the same kernel is correct for any extent, and the occupancy is chosen once.
    ** Buffers arrive as separate parameters rather than an array, which keeps the indirection out of
    ** the inner loop and lets the compiler see which pointers alias.
    */
    [[nodiscard]] std::string fused_codegen(const mag_fuse_graph_t &g) {
      bool written[MAG_FUSE_MAX_BUF] = {};
      for (uint8_t s = 0; s < g.num_stores; ++s) written[g.stores[s].buf] = true;
      const bool narrow = g.dtype != MAG_DTYPE_FLOAT32;

      /*
      ** Conversions are written as the two PTX instructions the hardware has, rather than through
      ** cuda_fp16.h. NVRTC only sees headers it is handed, and finding the toolkit's include
      ** directory at runtime is a worse dependency than two lines of assembly that cannot drift.
      ** cvt.rn.f16.f32 rounds to nearest even, which is what the eager kernels do and what the CPU
      ** backend does, so the three agree.
      */
      std::string src;
      if (narrow) {
        src +=
          "typedef unsigned short mag_st_t;\n"
          "__device__ __forceinline__ float mag_ld(mag_st_t h) {\n"
          "  float f; asm(\"cvt.f32.f16 %0, %1;\" : \"=f\"(f) : \"h\"(h)); return f;\n"
          "}\n"
          "__device__ __forceinline__ mag_st_t mag_st(float f) {\n"
          "  mag_st_t h; asm(\"cvt.rn.f16.f32 %0, %1;\" : \"=h\"(h) : \"f\"(f)); return h;\n"
          "}\n"
          "#define mag_rt(x) mag_ld(mag_st(x))\n\n";
      } else {
        src +=
          "typedef float mag_st_t;\n"
          "#define mag_ld(x) (x)\n"
          "#define mag_st(x) (x)\n"
          "#define mag_rt(x) (x)\n\n";
      }
      src += "extern \"C\" __global__ void mag_fused(";
      for (uint8_t b = 0; b < g.num_bufs; ++b) {
        /* Only operands the chain never writes get __restrict__, since a written buffer may be the
           same memory as one that is read. */
        src += written[b] ? "mag_st_t *b" : "const mag_st_t *__restrict__ b";
        src += std::to_string(b);
        src += ", ";
      }
      for (uint8_t k = 0; k < g.num_imms; ++k) {
        src += "float s";
        src += std::to_string(k);
        src += "_in, ";
      }
      src += "long long n) {\n";
      /* An immediate is a scalar of the storage type, so it rounds once, exactly as the operand it
         stands in for would have. */
      for (uint8_t k = 0; k < g.num_imms; ++k)
        src += "  const float s" + std::to_string(k) + " = mag_rt(s" + std::to_string(k) + "_in);\n";
      src += "  long long stride = (long long)gridDim.x * blockDim.x;\n";
      src += "  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {\n";

      char name[32];
      for (uint32_t i = 0; i < g.num_ins; ++i) {
        const mag_fuse_ins_t &ins = g.ins[i];
        if (ins.op == MAG_FUSE_OP_LOAD) {
          operand_name(ins.in[0], name, sizeof(name));
          /* The operand name already widens, in whichever position it appears. */
          src += "    const float r" + std::to_string(i) + " = " + name + ";\n";
          continue;
        }
        const char *form = fused_form(ins.op);
        src += "    const float r" + std::to_string(i) + " = mag_rt(";
        for (const char *p = form; *p; ++p) {
          if (*p == '$' && p[1] >= '0' && p[1] <= '2') {
            operand_name(ins.in[p[1] - '0'], name, sizeof(name));
            src += name;
            ++p;
          } else {
            src += *p;
          }
        }
        src += ");\n";
      }
      for (uint8_t s = 0; s < g.num_stores; ++s)
        src += "    b" + std::to_string(g.stores[s].buf) + "[i] = mag_st(r" + std::to_string(g.stores[s].reg) + ");\n";
      src += "  }\n}\n";
      return src;
    }

    struct fused_kernel final {
      CUmodule mod = nullptr;
      CUfunction fn = nullptr;
    };

    /* One cache per device, keyed on the graph's structure rather than its constants. */
    struct fused_cache final {
      std::mutex lock;
      std::unordered_map<uint64_t, fused_kernel> kernels;
      bool unavailable = false; /* Latched once NVRTC has been shown not to work here. */
    };

    fused_cache &cache_for(int ordinal) {
      static std::mutex reg_lock;
      static std::unordered_map<int, fused_cache *> reg;
      std::lock_guard guard {reg_lock};
      auto it = reg.find(ordinal);
      if (it == reg.end()) it = reg.emplace(ordinal, new fused_cache).first;
      return *it->second;
    }

    /*
    ** Compile to PTX for the architecture actually in the machine.
    **
    ** --fmad=false is load bearing. Left on, the compiler contracts a multiply and an add into one
    ** FMA, which rounds once where the eager kernels round twice, and the chain stops agreeing with
    ** the operators it replaces.
    */
    [[nodiscard]] bool compile_ptx(const std::string &src, int cc_major, int cc_minor, std::string &ptx, std::string &log) {
      nvrtcProgram prog = nullptr;
      if (nvrtcCreateProgram(&prog, src.c_str(), "mag_fused.cu", 0, nullptr, nullptr) != NVRTC_SUCCESS) return false;
      char arch[64];
      snprintf(arch, sizeof(arch), "--gpu-architecture=compute_%d%d", cc_major, cc_minor);
      const char *opts[] = {arch, "--fmad=false"};
      nvrtcResult rc = nvrtcCompileProgram(prog, static_cast<int>(std::size(opts)), opts);
      size_t log_size = 0;
      if (nvrtcGetProgramLogSize(prog, &log_size) == NVRTC_SUCCESS && log_size > 1) {
        log.resize(log_size);
        nvrtcGetProgramLog(prog, log.data());
      }
      if (rc != NVRTC_SUCCESS) { nvrtcDestroyProgram(&prog); return false; }
      size_t ptx_size = 0;
      if (nvrtcGetPTXSize(prog, &ptx_size) != NVRTC_SUCCESS) { nvrtcDestroyProgram(&prog); return false; }
      ptx.resize(ptx_size);
      bool ok = nvrtcGetPTX(prog, ptx.data()) == NVRTC_SUCCESS;
      nvrtcDestroyProgram(&prog);
      return ok;
    }
  }

  void fused_cache_shutdown(int ordinal) noexcept {
    fused_cache &c = cache_for(ordinal);
    std::lock_guard guard {c.lock};
    for (auto &[key, k] : c.kernels)
      if (k.mod) cuModuleUnload(k.mod);
    c.kernels.clear();
  }

  mag_status_t fused_op(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream) {
    if (mag_unlikely(!cmd.params))
      return mag_set_error(err, MAG_ERR_KERNEL, "cuda: fused command carries no graph.");
    const mag_fuse_graph_t *g = cmd.params->fused.graph;
    if (mag_unlikely(!g || !fused_supported(*g)))
      return mag_set_error(err, MAG_ERR_KERNEL, "cuda: no lowering for this fused chain.");
    if (mag_unlikely(cmd.num_in > MAG_FUSE_MAX_BUF))
      return mag_set_error(err, MAG_ERR_KERNEL, "cuda: fused chain binds more buffers than the graph allows.");

    int ordinal = 0;
    mag_cu_rt_check(err, cudaGetDevice(&ordinal), "failed to read the active device");
    fused_cache &c = cache_for(ordinal);

    uint64_t key = mag_fuse_graph_hash(g);
    CUfunction fn = nullptr;
    {
      std::lock_guard guard {c.lock};
      if (c.unavailable)
        return mag_set_error(err, MAG_ERR_KERNEL, "cuda: no runtime compiler for fused chains.");
      if (auto it = c.kernels.find(key); it != c.kernels.end()) {
        fn = it->second.fn;
      } else {
        int cc_major = 0, cc_minor = 0;
        mag_cu_rt_check(err, cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, ordinal), "failed to read compute capability");
        mag_cu_rt_check(err, cudaDeviceGetAttribute(&cc_minor, cudaDevAttrComputeCapabilityMinor, ordinal), "failed to read compute capability");
        std::string src = fused_codegen(*g), ptx, log;
        if (!compile_ptx(src, cc_major, cc_minor, ptx, log)) {
          /* Not an error the caller should see: core replays the chain one operator at a time. Say
             it happened, or the only symptom is that everything is quietly slower. */
          mag_log_warn("cuda: could not compile a fused chain; running it eagerly instead. %s", log.c_str());
          c.unavailable = true;
          return mag_set_error(err, MAG_ERR_KERNEL, "cuda: fused chain could not be compiled.");
        }
        fused_kernel k {};
        mag_cu_check(err, cuModuleLoadData(&k.mod, ptx.c_str()), "failed to load a compiled fused chain");
        mag_cu_check(err, cuModuleGetFunction(&k.fn, k.mod, "mag_fused"), "compiled fused chain has no entry point");
        c.kernels.emplace(key, k);
        fn = k.fn;
      }
    }

    /* Every operand is either the full chain length or one broadcast element, so the longest is the
       extent. Shape never enters the graph; this is the only extent there is. */
    /* Write access is asked for only where the chain writes: an operand may be a tensor borrowing
       memory it is not allowed to write, and asking for a mutable pointer to one aborts. */
    bool written[MAG_FUSE_MAX_BUF] = {};
    for (uint8_t st = 0; st < g->num_stores; ++st) written[g->stores[st].buf] = true;

    int64_t n = 0;
    void *bufs[MAG_FUSE_MAX_BUF] = {};
    for (uint32_t b = 0; b < cmd.num_in; ++b) {
      bufs[b] = reinterpret_cast<void *>(written[b] ? mag_tensor_data_ptr_mut(cmd.in[b])
                                                    : mag_tensor_data_ptr(cmd.in[b]));
      n = std::max<int64_t>(n, cmd.in[b]->meta.numel);
    }
    float imms[MAG_FUSE_MAX_IMM] = {}; /* Immediates arrive at call time; no chain uses them yet. */

    /* Kernel arguments are addresses of the values, in declaration order. */
    void *args[MAG_FUSE_MAX_BUF + MAG_FUSE_MAX_IMM + 1] = {};
    size_t argc = 0;
    for (uint8_t b = 0; b < g->num_bufs; ++b) args[argc++] = &bufs[b];
    for (uint8_t k = 0; k < g->num_imms; ++k) args[argc++] = &imms[k];
    args[argc++] = &n;

    auto blocks = static_cast<unsigned>(std::min<int64_t>((n + FUSED_BLOCK_SIZE - 1)/FUSED_BLOCK_SIZE,
                                                          std::numeric_limits<int>::max()));
    if (!blocks) return MAG_OK;
    mag_cu_check(err, cuLaunchKernel(fn, blocks, 1, 1, FUSED_BLOCK_SIZE, 1, 1, 0, stream, args, nullptr),
                 "failed to launch a fused chain");
    return MAG_OK;
  }
}
