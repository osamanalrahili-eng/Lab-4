#include "fpga_gemm_backend.h"

#include <vector>
#include <string>
#include <iostream>
#include <stdexcept>
#include <cstring>
#include <cmath>
#include <sys/time.h>
#include <iomanip>

#define FPGA_ENABLED 1

#if FPGA_ENABLED
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"
using namespace aocl_utils;
#endif

namespace {

double now_ms() {
    struct timeval tv;
    gettimeofday(&tv, 0);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

struct GemmStats {
    int compare_calls;
    int cpu_successes;
    int fpga_successes;
    int match_count;

    double cpu_total_ms;
    double fpga_total_ms;

    GemmStats()
        : compare_calls(0),
          cpu_successes(0),
          fpga_successes(0),
          match_count(0),
          cpu_total_ms(0.0),
          fpga_total_ms(0.0) {}
};

GemmStats g_stats;

bool validate_shapes(const std::vector<float>& A,
                     const std::vector<float>& B,
                     int M,
                     int K,
                     int N) {
    if (M <= 0 || K <= 0 || N <= 0) {
        std::cerr << "gemm_backend: invalid GEMM shape M=" << M
                  << " K=" << K << " N=" << N << std::endl;
        return false;
    }

    if ((int)A.size() != M * K) {
        std::cerr << "gemm_backend: A size mismatch. Expected "
                  << (M * K) << " got " << A.size() << std::endl;
        return false;
    }

    if ((int)B.size() != K * N) {
        std::cerr << "gemm_backend: B size mismatch. Expected "
                  << (K * N) << " got " << B.size() << std::endl;
        return false;
    }

    return true;
}

void gemm_cpu_impl(const std::vector<float>& A,
                   const std::vector<float>& B,
                   std::vector<float>& C,
                   int M,
                   int K,
                   int N) {
    if ((int)A.size() != M * K) {
        throw std::runtime_error("gemm_cpu_impl: A size mismatch");
    }
    if ((int)B.size() != K * N) {
        throw std::runtime_error("gemm_cpu_impl: B size mismatch");
    }

    C.assign(M * N, 0.0f);

    int i, j, k;
    for (i = 0; i < M; ++i) {
        for (j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

void compute_diff_stats(const std::vector<float>& A,
                        const std::vector<float>& B,
                        float& max_abs_diff,
                        float& avg_abs_diff) {
    max_abs_diff = 0.0f;
    avg_abs_diff = 0.0f;

    if (A.size() != B.size() || A.empty()) {
        max_abs_diff = -1.0f;
        avg_abs_diff = -1.0f;
        return;
    }

    double sum = 0.0;
    size_t i;
    for (i = 0; i < A.size(); ++i) {
        float d = std::fabs(A[i] - B[i]);
        if (d > max_abs_diff) {
            max_abs_diff = d;
        }
        sum += d;
    }

    avg_abs_diff = (float)(sum / (double)A.size());
}

void log_compare(const GemmCompareInfo& info) {
    std::cout << "[GEMM_COMPARE] "
              << "tag=" << info.tag
              << " M=" << info.M
              << " K=" << info.K
              << " N=" << info.N
              << " cpu_ok=" << (info.cpu_success ? 1 : 0)
              << " fpga_ok=" << (info.fpga_success ? 1 : 0)
              << " cpu_ms=" << std::fixed << std::setprecision(3) << info.cpu_time_ms
              << " fpga_ms=" << std::fixed << std::setprecision(3) << info.fpga_time_ms
              << " max_abs_diff=" << info.max_abs_diff
              << " avg_abs_diff=" << info.avg_abs_diff
              << " match=" << (info.outputs_match ? 1 : 0)
              << " return=" << (info.returned_fpga ? "fpga" : "cpu")
              << std::endl;
}

} // anonymous namespace

#if FPGA_ENABLED

static bool g_fpga_initialized = false;
static bool g_fpga_ready = false;

static cl_platform_id g_platform = 0;
static cl_device_id g_device = 0;
static cl_context g_context = 0;
static cl_command_queue g_queue = 0;
static cl_program g_program = 0;
static cl_kernel g_kernel = 0;

bool gemm_backend_init(const char* bitstream_path) {
    cl_int status;

    g_kernel = NULL;
    g_queue = NULL;
    g_program = NULL;
    g_context = NULL;
    g_platform = NULL;
    g_device = NULL;

    if (!setCwdToExeDir()) {
        gemm_backend_cleanup();
        return false;
    }


    g_platform = findPlatform("Intel(R) FPGA");
    if (g_platform == NULL) {
        std::cerr << "gemm_backend_init: failed to find Intel(R) FPGA platform" << std::endl;
        gemm_backend_cleanup();
        return false;
    }


    status = clGetDeviceIDs(g_platform, CL_DEVICE_TYPE_ALL, 1, &g_device, NULL);
    if (status != CL_SUCCESS) {
        std::cerr << "gemm_backend_init: clGetDeviceIDs failed, status = " << status << std::endl;
        gemm_backend_cleanup();
        return false;
    }

    g_context = clCreateContext(0, 1, &g_device, &oclContextCallback, NULL, &status);
    if (status != CL_SUCCESS) {
        std::cerr << "gemm_backend_init: clCreateContext failed, status = " << status << std::endl;
        gemm_backend_cleanup();
        return false;
    }

    g_queue = clCreateCommandQueue(g_context, g_device, CL_QUEUE_PROFILING_ENABLE, &status);
    if (status != CL_SUCCESS) {
        std::cerr << "gemm_backend_init: clCreateCommandQueue failed, status = " << status << std::endl;
        gemm_backend_cleanup();
        return false;
    }

    std::string binary_file = getBoardBinaryFile(bitstream_path, g_device);
    std::cout << "Using AOCX: " << binary_file << std::endl;

    g_program = createProgramFromBinary(g_context, binary_file.c_str(), &g_device, 1);
    if (g_program == NULL) {
        std::cerr << "gemm_backend_init: createProgramFromBinary failed" << std::endl;
        gemm_backend_cleanup();
        return false;
    }

    status = clBuildProgram(g_program, 1, &g_device, "", NULL, NULL);
    if (status != CL_SUCCESS) {
        std::cerr << "gemm_backend_init: clBuildProgram failed, status = " << status << std::endl;
        gemm_backend_cleanup();
        return false;
    }

    g_kernel = clCreateKernel(g_program, "gemm_kernel", &status);
    if (status != CL_SUCCESS) {
        std::cerr << "gemm_backend_init: clCreateKernel failed, status = " << status << std::endl;
        gemm_backend_cleanup();
        return false;
    }

    g_fpga_ready = true;
    g_fpga_initialized = true;
    return true;
}

void gemm_backend_cleanup() {
    if (g_kernel) {
        clReleaseKernel(g_kernel);
        g_kernel = 0;
    }
    if (g_program) {
        clReleaseProgram(g_program);
        g_program = 0;
    }
    if (g_queue) {
        clReleaseCommandQueue(g_queue);
        g_queue = 0;
    }
    if (g_context) {
        clReleaseContext(g_context);
        g_context = 0;
    }

    g_platform = 0;
    g_device = 0;
    g_fpga_ready = false;
    g_fpga_initialized = false;
}

bool gemm_backend_is_fpga_ready() {
    return g_fpga_ready;
}

bool gemm_backend_run_fpga(const std::vector<float>& A,
                           const std::vector<float>& B,
                           std::vector<float>& C,
                           int M,
                           int K,
                           int N) {
    if (!g_fpga_ready) {
        std::cerr << "gemm_backend_run_fpga: FPGA backend not ready" << std::endl;
        return false;
    }

    if (!validate_shapes(A, B, M, K, N)) {
        return false;
    }

    cl_int status = CL_SUCCESS;
    cl_mem buf_A = 0;
    cl_mem buf_B = 0;
    cl_mem buf_C = 0;

    const size_t bytes_A = (size_t)M * K * sizeof(float);
    const size_t bytes_B = (size_t)K * N * sizeof(float);
    const size_t bytes_C = (size_t)M * N * sizeof(float);

    C.assign(M * N, 0.0f);

    buf_A = clCreateBuffer(g_context, CL_MEM_READ_ONLY, bytes_A, 0, &status);
    if (status != CL_SUCCESS || !buf_A) {
        std::cerr << "gemm_backend_run_fpga: clCreateBuffer A failed, status = "
                  << status << std::endl;
        return false;
    }

    // TODO 1: create buffer for B and C
    buf_B = clCreateBuffer(g_context, CL_MEM_READ_ONLY, bytes_B, 0, &status);
    if (status != CL_SUCCESS || !buf_B) {
        std::cerr << "gemm_backend_run_fpga: clCreateBuffer B failed, status = "
                  << status << std::endl;
        clReleaseMemObject(buf_A);
        return false;
    }

    buf_C = clCreateBuffer(g_context, CL_MEM_WRITE_ONLY, bytes_C, 0, &status);
    if (status != CL_SUCCESS || !buf_C) {
        std::cerr << "gemm_backend_run_fpga: clCreateBuffer C failed, status = "
                  << status << std::endl;
        clReleaseMemObject(buf_A);
        clReleaseMemObject(buf_B);
        return false;
    }

    status = clEnqueueWriteBuffer(g_queue, buf_A, CL_TRUE, 0, bytes_A,
                                  (const void*)A.data(), 0, 0, 0);
    if (status != CL_SUCCESS) {
        std::cerr << "gemm_backend_run_fpga: write A failed, status = "
                  << status << std::endl;
        clReleaseMemObject(buf_A);
        clReleaseMemObject(buf_B);
        clReleaseMemObject(buf_C);
        return false;
    }

    // TODO 2: write to buffer for B
    status = clEnqueueWriteBuffer(g_queue, buf_B, CL_TRUE, 0, bytes_B,
                                  (const void*)B.data(), 0, 0, 0);
    if (status != CL_SUCCESS) {
        std::cerr << "gemm_backend_run_fpga: write B failed, status = "
                  << status << std::endl;
        clReleaseMemObject(buf_A);
        clReleaseMemObject(buf_B);
        clReleaseMemObject(buf_C);
        return false;
    }

    // TODO 3: set kernel arguments
    status  = clSetKernelArg(g_kernel, 0, sizeof(cl_mem), &buf_A);
    status |= clSetKernelArg(g_kernel, 1, sizeof(cl_mem), &buf_B);
    status |= clSetKernelArg(g_kernel, 2, sizeof(cl_mem), &buf_C);
    status |= clSetKernelArg(g_kernel, 3, sizeof(int), &M);
    status |= clSetKernelArg(g_kernel, 4, sizeof(int), &K);
    status |= clSetKernelArg(g_kernel, 5, sizeof(int), &N);

    if (status != CL_SUCCESS) {
        std::cerr << "gemm_backend_run_fpga: clSetKernelArg failed, status = "
                  << status << std::endl;
        clReleaseMemObject(buf_A);
        clReleaseMemObject(buf_B);
        clReleaseMemObject(buf_C);
        return false;
    }

    size_t global_work_size[2];
    global_work_size[0] = (size_t)M;
    global_work_size[1] = (size_t)N;

    status = clEnqueueNDRangeKernel(g_queue,
                                    g_kernel,
                                    2,
                                    0,
                                    global_work_size,
                                    0,
                                    0,
                                    0,
                                    0);
    if (status != CL_SUCCESS) {
        std::cerr << "gemm_backend_run_fpga: clEnqueueNDRangeKernel failed, status = "
                  << status << std::endl;
        clReleaseMemObject(buf_A);
        clReleaseMemObject(buf_B);
        clReleaseMemObject(buf_C);
        return false;
    }

    status = clFinish(g_queue);
    if (status != CL_SUCCESS) {
        std::cerr << "gemm_backend_run_fpga: clFinish failed, status = "
                  << status << std::endl;
        clReleaseMemObject(buf_A);
        clReleaseMemObject(buf_B);
        clReleaseMemObject(buf_C);
        return false;
    }

    // TODO 4: Read output C data
    status = clEnqueueReadBuffer(g_queue, buf_C, CL_TRUE, 0, bytes_C,
                                 (void*)C.data(), 0, 0, 0);
    if (status != CL_SUCCESS) {
        std::cerr << "gemm_backend_run_fpga: read C failed, status = "
                  << status << std::endl;
        clReleaseMemObject(buf_A);
        clReleaseMemObject(buf_B);
        clReleaseMemObject(buf_C);
        return false;
    }

    clReleaseMemObject(buf_A);
    clReleaseMemObject(buf_B);
    clReleaseMemObject(buf_C);

    return true;
}

#else

static bool g_fpga_ready = false;

bool gemm_backend_init(const char* bitstream_path) {
    (void)bitstream_path;
    g_fpga_ready = false;

    status = clGetPlatformIDs(0, NULL, &num_platforms);
    if (status != CL_SUCCESS || num_platforms == 0) {
        std::cerr << "[FPGA] No OpenCL platform found. Check runtime environment." << std::endl;
        return false;
    }
    return false;
}

void gemm_backend_cleanup() {
    g_fpga_ready = false;
}

bool gemm_backend_is_fpga_ready() {
    return false;
}

bool gemm_backend_run_fpga(const std::vector<float>& A,
                           const std::vector<float>& B,
                           std::vector<float>& C,
                           int M,
                           int K,
                           int N) {
    (void)A;
    (void)B;
    (void)C;
    (void)M;
    (void)K;
    (void)N;
    std::cerr << "gemm_backend_run_fpga: FPGA path disabled" << std::endl;
    return false;
}

#endif

void gemm_backend_run_cpu(const std::vector<float>& A,
                          const std::vector<float>& B,
                          std::vector<float>& C,
                          int M,
                          int K,
                          int N) {
    if (!validate_shapes(A, B, M, K, N)) {
        throw std::runtime_error("gemm_backend_run_cpu: invalid GEMM inputs");
    }

    gemm_cpu_impl(A, B, C, M, K, N);
}

bool gemm_backend_run_both_compare(const std::vector<float>& A,
                                   const std::vector<float>& B,
                                   std::vector<float>& C_out,
                                   int M,
                                   int K,
                                   int N,
                                   const char* tag,
                                   GemmCompareInfo* info) {
    GemmCompareInfo local;
    local.tag = (tag ? tag : "unnamed");
    local.M = M;
    local.K = K;
    local.N = N;
    local.cpu_success = false;
    local.fpga_success = false;
    local.cpu_time_ms = 0.0;
    local.fpga_time_ms = 0.0;
    local.max_abs_diff = -1.0f;
    local.avg_abs_diff = -1.0f;
    local.outputs_match = false;
    local.returned_fpga = false;

    std::vector<float> C_cpu;
    std::vector<float> C_fpga;

    g_stats.compare_calls += 1;

    {
        double t0 = now_ms();
        try {
            gemm_backend_run_cpu(A, B, C_cpu, M, K, N);
            local.cpu_success = true;
        } catch (const std::exception& e) {
            std::cerr << "CPU GEMM failed for tag=" << local.tag
                      << " : " << e.what() << std::endl;
            local.cpu_success = false;
        }
        double t1 = now_ms();
        local.cpu_time_ms = t1 - t0;

        if (local.cpu_success) {
            g_stats.cpu_successes += 1;
            g_stats.cpu_total_ms += local.cpu_time_ms;
        }
    }

    {
        double t0 = now_ms();
        local.fpga_success = gemm_backend_run_fpga(A, B, C_fpga, M, K, N);
        double t1 = now_ms();
        local.fpga_time_ms = t1 - t0;

        if (local.fpga_success) {
            g_stats.fpga_successes += 1;
            g_stats.fpga_total_ms += local.fpga_time_ms;
        }
    }

    if (local.cpu_success && local.fpga_success) {
        compute_diff_stats(C_cpu, C_fpga, local.max_abs_diff, local.avg_abs_diff);
        local.outputs_match =
            (local.max_abs_diff >= 0.0f && local.max_abs_diff < 1e-3f);

        if (local.outputs_match) {
            g_stats.match_count += 1;
        }
    }

    if (local.fpga_success && local.outputs_match) {
        C_out = C_fpga;
        local.returned_fpga = true;
    } else if (local.cpu_success) {
        C_out = C_cpu;
        local.returned_fpga = false;
    } else if (local.fpga_success) {
        C_out = C_fpga;
        local.returned_fpga = true;
    } else {
        log_compare(local);
        if (info) {
            *info = local;
        }
        return false;
    }

    log_compare(local);
    if (info) {
        *info = local;
    }
    return true;
}

void gemm_backend_print_stats() {
    std::cout << "\n=== GEMM Compare Summary ===" << std::endl;
    std::cout << "compare calls   : " << g_stats.compare_calls << std::endl;
    std::cout << "cpu successes   : " << g_stats.cpu_successes << std::endl;
    std::cout << "fpga successes  : " << g_stats.fpga_successes << std::endl;
    std::cout << "matches         : " << g_stats.match_count << std::endl;

    std::cout << "cpu total ms    : " << std::fixed << std::setprecision(3)
              << g_stats.cpu_total_ms << std::endl;
    if (g_stats.cpu_successes > 0) {
        std::cout << "cpu avg ms      : "
                  << (g_stats.cpu_total_ms / g_stats.cpu_successes) << std::endl;
    }

    std::cout << "fpga total ms   : " << std::fixed << std::setprecision(3)
              << g_stats.fpga_total_ms << std::endl;
    if (g_stats.fpga_successes > 0) {
        std::cout << "fpga avg ms     : "
                  << (g_stats.fpga_total_ms / g_stats.fpga_successes) << std::endl;
    }

    std::cout << "============================\n" << std::endl;
}