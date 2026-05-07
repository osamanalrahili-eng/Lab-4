#include <chrono>
#include <exception>
#include <iostream>
#include <string>
#include <thread>

#include <unistd.h>
#include "capture_digit.h"
#include "digit_cnn.h"
#include "fpga_gemm_backend.h"

void cleanup() {
    gemm_backend_cleanup();
}

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            std::cerr << "Usage: " << argv[0] << " <weight_dir> [max_iters]\n";
            return 1;
        }

        const std::string weight_dir = argv[1];
        const int max_iters = (argc >= 3) ? std::stoi(argv[2]) : -1;

        std::cout << "Loading CNN weights from: " << weight_dir << "\n";
        SimpleCNN64 model = load_model_from_bin_dir(weight_dir);
        std::cout << "Model loaded successfully.\n";

        // ADD FPGA INIT
        bool fpga_ok = gemm_backend_init("gemm_kernel");  
        std::cout << "FPGA ready = " << (fpga_ok ? "yes" : "no") << "\n";

        CaptureDevice cap;
        if (!init_capture_device(cap, 160, 70, 6)) {
            std::cerr << "Failed to initialize capture device.\n";
            return 1;
        }

        std::cout << "Capture initialized.\n";
        std::cout << "Press FPGA button to capture one ROI and run inference.\n";
        std::cout << "Loop starts now. Ctrl+C to exit.\n";

        int iter = 0;
        while (max_iters < 0 || iter < max_iters) {
            std::cout << "\n[Loop " << iter << "] Waiting for capture...\n";
            GrayImage gray = capture_roi_gray_on_button(cap, false, "capture");
            std::cout << "Captured ROI: " << gray.width << "x" << gray.height << "\n";

            DigitPrediction result = predict_digit_from_gray(
                model,
                gray,
                5,
                48,
                5,
                false,
                false,
                "final_canvas.pgm"
            );

            print_vector(result.log_probs, "LogSoftmax output");
            std::cout << "Predicted digit: " << result.pred << "\n";
            display_digit_ledr(cap, result.pred);

            ++iter;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        shutdown_capture_device(cap);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
