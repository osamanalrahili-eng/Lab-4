#include "digit_cnn.h"
#include "fpga_gemm_backend.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

Tensor3D::Tensor3D() : C(0), H(0), W(0) {}
Tensor3D::Tensor3D(int c, int h, int w, float val)
    : C(c), H(h), W(w), data(static_cast<size_t>(c) * h * w, val) {}


// -----------------------------------------------------------------------------
// Access element at (channel, y, x) in a 3D tensor.
//
// Data is stored in a flat vector in row-major order:
//
//   index = (c * H + y) * W + x
//
// -----------------------------------------------------------------------------

float& Tensor3D::at(int c, int y, int x) {
    return data[(static_cast<size_t>(c) * H + y) * W + x];
}


const float& Tensor3D::at(int c, int y, int x) const {
    return data[(static_cast<size_t>(c) * H + y) * W + x];
}

Vector1D::Vector1D() : N(0) {}
Vector1D::Vector1D(int n, float val) : N(n), data(static_cast<size_t>(n), val) {}




// -----------------------------------------------------------------------------
// Access element at index i in a 1D vector.
//
// This is a simple wrapper over std::vector for consistency with Tensor3D.
// -----------------------------------------------------------------------------
float& Vector1D::at(int i) { return data[static_cast<size_t>(i)]; }
const float& Vector1D::at(int i) const { return data[static_cast<size_t>(i)]; }

ConvLayer::ConvLayer()
    : out_channels(0), in_channels(0), kernel_h(0), kernel_w(0), padding(0) {}


// -----------------------------------------------------------------------------
// Access weight for convolution filter.
//
// Dimensions:
//   weight[oc][ic][ky][kx]
//
// Stored in flattened format.
//
// STUDENT NOTE:
//   Understand how 4D filter weights are stored in 1D memory.
// -----------------------------------------------------------------------------
float& ConvLayer::w(int oc, int ic, int ky, int kx) {
    size_t idx = (((static_cast<size_t>(oc) * in_channels + ic) * kernel_h + ky) * kernel_w + kx);
    return weight[idx];
}

const float& ConvLayer::w(int oc, int ic, int ky, int kx) const {
    size_t idx = (((static_cast<size_t>(oc) * in_channels + ic) * kernel_h + ky) * kernel_w + kx);
    return weight[idx];
}

LinearLayer::LinearLayer() : out_features(0), in_features(0) {}
float& LinearLayer::w(int o, int i) { return weight[static_cast<size_t>(o) * in_features + i]; }

// -----------------------------------------------------------------------------
// Access weight for fully connected layer.
//
// Dimensions:
//   weight[out_feature][in_feature]
//
// Stored as a flattened 2D matrix.
//
// STUDENT NOTE:
//   This is equivalent to a matrix multiplication:
//   output = W × input
// -----------------------------------------------------------------------------
const float& LinearLayer::w(int o, int i) const { return weight[static_cast<size_t>(o) * in_features + i]; }


// -----------------------------------------------------------------------------
// Convolution using im2col + GEMM
// -----------------------------------------------------------------------------
static Tensor3D conv2d_im2col_gemm(const Tensor3D& input, const ConvLayer& layer) {
    if (input.C != layer.in_channels) {
        throw std::runtime_error("conv2d_im2col_gemm: channel mismatch");
    }

    const int C_in = layer.in_channels;
    const int C_out = layer.out_channels;
    const int H = input.H;
    const int W = input.W;
    const int K = layer.kernel_h;
    const int pad = layer.padding;

    const int HW = H * W;
    const int KKK = C_in * K * K;

    std::vector<float> A(static_cast<size_t>(HW) * KKK, 0.0f);
    for (int oy = 0; oy < H; ++oy) {
        for (int ox = 0; ox < W; ++ox) {
            const int row = oy * W + ox;
            for (int ic = 0; ic < C_in; ++ic) {
                for (int ky = 0; ky < K; ++ky) {
                    for (int kx = 0; kx < K; ++kx) {
                        const int iy = oy + ky - pad;
                        const int ix = ox + kx - pad;
                        float val = 0.0f;
                        if (iy >= 0 && iy < H && ix >= 0 && ix < W) {
                            val = input.at(ic, iy, ix);
                        }
                        const int col = ic * (K * K) + ky * K + kx;
                        A[static_cast<size_t>(row) * KKK + col] = val;
                    }
                }
            }
        }
    }

    std::vector<float> B(static_cast<size_t>(KKK) * C_out, 0.0f);
    for (int oc = 0; oc < C_out; ++oc) {
        for (int ic = 0; ic < C_in; ++ic) {
            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {
                    const int row = ic * (K * K) + ky * K + kx;
                    B[static_cast<size_t>(row) * C_out + oc] = layer.w(oc, ic, ky, kx);
                }
            }
        }
    }

    // std::vector<float> Cmat(static_cast<size_t>(HW) * C_out, 0.0f);
    // for (int i = 0; i < HW; ++i) {
    //     for (int j = 0; j < C_out; ++j) {
    //         float sum = layer.bias[static_cast<size_t>(j)];
    //         for (int k = 0; k < KKK; ++k) {
    //             sum += A[static_cast<size_t>(i) * KKK + k] * B[static_cast<size_t>(k) * C_out + j];
    //         }
    //         Cmat[static_cast<size_t>(i) * C_out + j] = sum;
    //     }
    // }

    std::vector<float> Cmat;
    GemmCompareInfo info;

    bool ok = gemm_backend_run_both_compare(A, B, Cmat, HW, KKK, C_out, "gemm_kernel", &info);
    if (!ok) {
        throw std::runtime_error("conv2d_im2col_gemm: both CPU and FPGA GEMM failed");
    }

    // bias add stays here
    for (int i = 0; i < HW; ++i) {
        for (int j = 0; j < C_out; ++j) {
            Cmat[static_cast<size_t>(i) * C_out + j] += layer.bias[static_cast<size_t>(j)];
        }
    }

    Tensor3D output(C_out, H, W, 0.0f);
    for (int oc = 0; oc < C_out; ++oc) {
        for (int oy = 0; oy < H; ++oy) {
            for (int ox = 0; ox < W; ++ox) {
                const int idx = (oy * W + ox) * C_out + oc;
                output.at(oc, oy, ox) = Cmat[static_cast<size_t>(idx)];
            }
        }
    }

    return output;
}


// -----------------------------------------------------------------------------
// ReLU activation function for Tensor3D.
//
// Applies:
//   output = max(0, input)
//
// STUDENT NOTE:
//   Introduces non-linearity into the network.
// -----------------------------------------------------------------------------
static Tensor3D relu(const Tensor3D& input) {
    Tensor3D output = input;
    for (float& v : output.data) {
        if (v < 0.0f) v = 0.0f;
    }
    return output;
}

// -----------------------------------------------------------------------------
// ReLU activation for 1D vector (used after fully connected layer).
// -----------------------------------------------------------------------------
static Vector1D relu(const Vector1D& input) {
    Vector1D output = input;
    for (float& v : output.data) {
        if (v < 0.0f) v = 0.0f;
    }
    return output;
}


// -----------------------------------------------------------------------------
// Max pooling with kernel size 2x2 and stride 2.
//
// Reduces spatial dimensions by half.
//
// For each 2x2 region, takes the maximum value.
//
// STUDENT NOTE:
//   Helps reduce computation and introduces translation invariance.
// -----------------------------------------------------------------------------
static Tensor3D maxpool2x2_stride2(const Tensor3D& input) {
    if (input.H % 2 != 0 || input.W % 2 != 0) {
        throw std::runtime_error("maxpool2x2_stride2: input H/W must be even");
    }

    Tensor3D output(input.C, input.H / 2, input.W / 2, 0.0f);
    for (int c = 0; c < input.C; ++c) {
        for (int oy = 0; oy < output.H; ++oy) {
            for (int ox = 0; ox < output.W; ++ox) {
                float m = -std::numeric_limits<float>::infinity();
                for (int ky = 0; ky < 2; ++ky) {
                    for (int kx = 0; kx < 2; ++kx) {
                        const int iy = oy * 2 + ky;
                        const int ix = ox * 2 + kx;
                        m = std::max(m, input.at(c, iy, ix));
                    }
                }
                output.at(c, oy, ox) = m;
            }
        }
    }
    return output;
}


// -----------------------------------------------------------------------------
// Flatten a 3D tensor [C, H, W] into a 1D vector.
//
// Required before passing data into fully connected layers.
//
// Ordering:
//   channel → row → column
//
// STUDENT TASK:
//   Verify how indices are mapped during flattening.
// -----------------------------------------------------------------------------
static Vector1D flatten(const Tensor3D& input) {
    Vector1D out(input.C * input.H * input.W, 0.0f);
    int idx = 0;
    for (int c = 0; c < input.C; ++c) {
        for (int y = 0; y < input.H; ++y) {
            for (int x = 0; x < input.W; ++x) {
                out.at(idx++) = input.at(c, y, x);
            }
        }
    }
    return out;
}


// -----------------------------------------------------------------------------
// Fully connected (linear) layer.
//
// Computes:
//   output[o] = sum(input[i] * weight[o][i]) + bias[o]
//
// Equivalent to matrix-vector multiplication.
//
// STUDENT NOTE:
//   This is the same as GEMM when batch size = 1.
// -----------------------------------------------------------------------------
static Vector1D linear(const Vector1D& input, const LinearLayer& layer) {
    if (input.N != layer.in_features) {
        throw std::runtime_error("linear: input feature mismatch");
    }

    Vector1D out(layer.out_features, 0.0f);
    for (int o = 0; o < layer.out_features; ++o) {
        float sum = layer.bias[static_cast<size_t>(o)];
        for (int i = 0; i < layer.in_features; ++i) {
            sum += input.at(i) * layer.w(o, i);
        }
        out.at(o) = sum;
    }
    return out;
}


// -----------------------------------------------------------------------------
// Log-Softmax function.
//
// Converts raw scores (logits) into log-probabilities.
//
// Uses numerically stable computation:
//   subtract max before exponentiation
//
// STUDENT NOTE:
//   Final output represents log probabilities of each digit.
// -----------------------------------------------------------------------------
static Vector1D log_softmax(const Vector1D& input) {
    Vector1D out(input.N, 0.0f);
    float max_v = input.at(0);
    for (int i = 1; i < input.N; ++i) {
        max_v = std::max(max_v, input.at(i));
    }

    double sum_exp = 0.0;
    for (int i = 0; i < input.N; ++i) {
        sum_exp += std::exp(static_cast<double>(input.at(i) - max_v));
    }

    const float log_sum_exp = max_v + static_cast<float>(std::log(sum_exp));
    for (int i = 0; i < input.N; ++i) {
        out.at(i) = input.at(i) - log_sum_exp;
    }
    return out;
}


// -----------------------------------------------------------------------------
// Reference implementation of 2D convolution with zero padding.
//
// This function performs convolution directly using nested loops.
// It is easy to understand but not optimized.
//
// PURPOSE IN LAB:
//   - Serves as the ground-truth implementation
//   - Used to verify correctness of im2col-based convolution
//
// STUDENT TASK:
//   Compare outputs of this function with conv2d_im2col_gemm()
// -----------------------------------------------------------------------------
Tensor3D conv2d_same_pad(const Tensor3D& input, const ConvLayer& layer) {
    if (input.C != layer.in_channels) {
        throw std::runtime_error("conv2d_same_pad: input channel mismatch");
    }

    Tensor3D output(layer.out_channels, input.H, input.W, 0.0f);

    for (int oc = 0; oc < layer.out_channels; ++oc) {
        for (int oy = 0; oy < input.H; ++oy) {
            for (int ox = 0; ox < input.W; ++ox) {
                float sum = layer.bias[static_cast<size_t>(oc)];

                for (int ic = 0; ic < layer.in_channels; ++ic) {
                    for (int ky = 0; ky < layer.kernel_h; ++ky) {
                        for (int kx = 0; kx < layer.kernel_w; ++kx) {
                            int iy = oy + ky - layer.padding;
                            int ix = ox + kx - layer.padding;

                            if (iy < 0 || iy >= input.H || ix < 0 || ix >= input.W) {
                                continue; // zero padding
                            }

                            sum += input.at(ic, iy, ix) * layer.w(oc, ic, ky, kx);
                        }
                    }
                }

                output.at(oc, oy, ox) = sum;
            }
        }
    }

    return output;
}


void compare_conv_implementations(const Tensor3D& x1, const Tensor3D& x2){
    float max_diff = 0.0f;
    for (size_t i = 0; i < x1.data.size(); i++) {
        max_diff = std::max(max_diff, std::abs(x1.data[i] - x2.data[i]));
    }
    std::cout << "Max diff: " << max_diff << std::endl;
}

// -----------------------------------------------------------------------------
// Forward pass of the CNN.
//
// Pipeline:
//   Conv → ReLU → Pool
//   Conv → ReLU → Pool
//   Conv → ReLU → Pool
//   Flatten → FC → ReLU → FC → LogSoftmax
//
// LAB STRUCTURE:
//   - x1: reference convolution
//   - x2: im2col-based convolution
//   - Students switch between them to validate correctness
//
// STUDENT TASK:
//   - Implement conv2d_im2col_gemm()
//   - Replace x = x1 with x = x2
//   - Ensure outputs match reference implementation
// -----------------------------------------------------------------------------

Vector1D forward(const SimpleCNN64& model, const Tensor3D& input) {
    if (input.C != 1 || input.H != 64 || input.W != 64) {
        throw std::runtime_error("forward: expected input shape [1,64,64]");
    }

    
    Tensor3D x1 = conv2d_same_pad(input, model.conv1);
    Tensor3D x2 = conv2d_im2col_gemm(input, model.conv1);
    Tensor3D x = x1; //TODO: change this to x = x2 when you finish implementing the conv2d_im2col_gemm
    compare_conv_implementations(x1,x2);
    x = relu(x);
    x = maxpool2x2_stride2(x);

    x1 = conv2d_same_pad(x, model.conv2);
    x2 = conv2d_im2col_gemm(x, model.conv2);
    x = x1; //TODO: change this to x = x2 when you finish implementing the conv2d_im2col_gemm
    compare_conv_implementations(x1,x2);
    x = relu(x);
    x = maxpool2x2_stride2(x);

    x1 = conv2d_same_pad(x, model.conv3);
    x2 = conv2d_im2col_gemm(x, model.conv3);
    x = x1; //TODO: change this to x = x2 when you finish implementing the conv2d_im2col_gemm
    compare_conv_implementations(x1,x2);
    x = x1;
    x = relu(x);
    x = maxpool2x2_stride2(x);

    Vector1D f = flatten(x);
    f = linear(f, model.fc1);
    f = relu(f);
    f = linear(f, model.fc2);
    return log_softmax(f);
}


// -----------------------------------------------------------------------------
// Returns index of maximum value in vector.
//
// Used to select predicted digit.
//
// STUDENT NOTE:
//   Final prediction = class with highest log probability.
// -----------------------------------------------------------------------------
int argmax(const Vector1D& v) {
    int best_idx = 0;
    float best_val = v.at(0);
    for (int i = 1; i < v.N; ++i) {
        if (v.at(i) > best_val) {
            best_val = v.at(i);
            best_idx = i;
        }
    }
    return best_idx;
}


// -----------------------------------------------------------------------------
// Load floating-point values from binary file.
//
// Used to load trained model weights.
//
// Includes size validation to ensure correctness.
//
// STUDENT NOTE:
//   Weights are exported from Python and loaded here for inference.
// -----------------------------------------------------------------------------
static void load_floats_from_bin(const std::string& path, std::vector<float>& buffer, size_t expected_count) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Could not open file: " + path);
    }

    in.seekg(0, std::ios::end);
    std::streamsize file_size = in.tellg();
    in.seekg(0, std::ios::beg);

    const std::streamsize expected_bytes = static_cast<std::streamsize>(expected_count * sizeof(float));
    if (file_size != expected_bytes) {
        throw std::runtime_error(
            "File size mismatch for " + path +
            " | expected bytes = " + std::to_string(expected_bytes) +
            " | actual bytes = " + std::to_string(file_size));
    }

    buffer.resize(expected_count);
    if (!in.read(reinterpret_cast<char*>(buffer.data()), expected_bytes)) {
        throw std::runtime_error("Failed to read file: " + path);
    }
}


// -----------------------------------------------------------------------------
// Define CNN architecture and allocate memory for weights.
//
// DOES NOT load actual values.
//
// STUDENT NOTE:
//   This defines the structure of the model used during training.
// -----------------------------------------------------------------------------
SimpleCNN64 create_model_structure() {
    SimpleCNN64 m;

    m.conv1.out_channels = 16;
    m.conv1.in_channels = 1;
    m.conv1.kernel_h = 3;
    m.conv1.kernel_w = 3;
    m.conv1.padding = 1;
    m.conv1.weight.resize(16 * 1 * 3 * 3);
    m.conv1.bias.resize(16);

    m.conv2.out_channels = 32;
    m.conv2.in_channels = 16;
    m.conv2.kernel_h = 3;
    m.conv2.kernel_w = 3;
    m.conv2.padding = 1;
    m.conv2.weight.resize(32 * 16 * 3 * 3);
    m.conv2.bias.resize(32);

    m.conv3.out_channels = 64;
    m.conv3.in_channels = 32;
    m.conv3.kernel_h = 3;
    m.conv3.kernel_w = 3;
    m.conv3.padding = 1;
    m.conv3.weight.resize(64 * 32 * 3 * 3);
    m.conv3.bias.resize(64);

    m.fc1.out_features = 128;
    m.fc1.in_features = 64 * 8 * 8;
    m.fc1.weight.resize(128 * (64 * 8 * 8));
    m.fc1.bias.resize(128);

    m.fc2.out_features = 10;
    m.fc2.in_features = 128;
    m.fc2.weight.resize(10 * 128);
    m.fc2.bias.resize(10);

    return m;
}


// -----------------------------------------------------------------------------
// Load trained weights from binary files into model.
//
// Each layer loads:
//   - weights
//   - bias
//
// STUDENT NOTE:
//   This connects Python-trained model → C++ inference.
// -----------------------------------------------------------------------------
SimpleCNN64 load_model_from_bin_dir(const std::string& dir_path) {
    SimpleCNN64 model = create_model_structure();

    load_floats_from_bin(dir_path + "/conv1_weight.bin", model.conv1.weight, 16 * 1 * 3 * 3);
    load_floats_from_bin(dir_path + "/conv1_bias.bin",   model.conv1.bias,   16);

    load_floats_from_bin(dir_path + "/conv2_weight.bin", model.conv2.weight, 32 * 16 * 3 * 3);
    load_floats_from_bin(dir_path + "/conv2_bias.bin",   model.conv2.bias,   32);

    load_floats_from_bin(dir_path + "/conv3_weight.bin", model.conv3.weight, 64 * 32 * 3 * 3);
    load_floats_from_bin(dir_path + "/conv3_bias.bin",   model.conv3.bias,   64);

    load_floats_from_bin(dir_path + "/fc1_weight.bin", model.fc1.weight, 128 * (64 * 8 * 8));
    load_floats_from_bin(dir_path + "/fc1_bias.bin",   model.fc1.bias,   128);

    load_floats_from_bin(dir_path + "/fc2_weight.bin", model.fc2.weight, 10 * 128);
    load_floats_from_bin(dir_path + "/fc2_bias.bin",   model.fc2.bias,   10);

    return model;
}


// -----------------------------------------------------------------------------
// Convert grayscale image to normalized tensor.
//
// Steps:
//   1. Convert pixel values from [0,255] → [0,1]
//   2. Apply normalization using mean and std
//
// IMPORTANT:
//   Must match training preprocessing EXACTLY.
//
// STUDENT TASK:
//   Observe effect of removing normalization.
// -----------------------------------------------------------------------------
Tensor3D grayimage_to_normalized_tensor(const GrayImage& img) {
    if (img.width != 64 || img.height != 64) {
        throw std::runtime_error("grayimage_to_normalized_tensor: expected 64x64 image");
    }

    Tensor3D out(1, 64, 64, 0.0f);
    const float mean = 0.1307f;
    const float stdv = 0.3081f;

    for (int y = 0; y < 64; ++y) {
        for (int x = 0; x < 64; ++x) {
            const float v01 = static_cast<float>(img.at(y, x)) / 255.0f;
            out.at(0, y, x) = (v01 - mean) / stdv;
        }
    }
    return out;
}


// -----------------------------------------------------------------------------
// Print vector values (useful for debugging).
// -----------------------------------------------------------------------------
void print_vector(const Vector1D& v, const std::string& name) {
    std::cout << name << ": ";
    for (int i = 0; i < v.N; ++i) {
        std::cout << v.at(i);
        if (i + 1 != v.N) std::cout << ' ';
    }
    std::cout << '\n';
}


// -----------------------------------------------------------------------------
// Full digit recognition pipeline.
//
// Flow:
//   Raw image
//      → preprocessing (centering, resizing)
//      → normalized tensor
//      → CNN forward pass
//      → predicted digit
//
// STUDENT NOTE:
//   This function connects:
//     image input → preprocessing → CNN → prediction
//
// This is the complete system flow.
// -----------------------------------------------------------------------------
DigitPrediction predict_digit_from_gray(
    const SimpleCNN64& model,
    const GrayImage& gray,
    int pad,
    int target_box_size,
    int tight_eps,
    bool verbose_preprocess,
    bool save_debug_canvas,
    const std::string& debug_canvas_path) {

    DigitPrediction result{};
    result.canvas64 = preprocess_roi_to_canvas_64(gray, pad, target_box_size, tight_eps, verbose_preprocess);

    if (save_debug_canvas) {
        save_pgm(result.canvas64, debug_canvas_path);
    }

    const Tensor3D input = grayimage_to_normalized_tensor(result.canvas64);
    result.log_probs = forward(model, input);
    result.pred = argmax(result.log_probs);
    return result;
}
