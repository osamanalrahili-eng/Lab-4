#include "capture_digit.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include "bmp_utility.h"

namespace {

constexpr std::uintptr_t HW_REGS_BASE = 0xff200000;
constexpr std::uintptr_t HW_REGS_SPAN = 0x00200000;
constexpr std::uintptr_t HW_REGS_MASK = HW_REGS_SPAN - 1;
constexpr std::uintptr_t PUSH_BASE    = 0x3010;
//constexpr std::uintptr_t VIDEO_BASE   = 0x3060;
constexpr std::uintptr_t VIDEO_BASE   = 0x0000;
constexpr std::uintptr_t LEDR_BASE  = 0x1000;
constexpr std::uintptr_t FPGA_ONCHIP_BASE = 0xC8000000;

constexpr int STRIDE_W = 512;
constexpr int CAP_W = 320;
constexpr int CAP_H = 240;
constexpr int IMAGE_SPAN = STRIDE_W * CAP_H * 2;

static unsigned char rgb565_to_gray(unsigned short p) {
    const unsigned int r5 = (p >> 11) & 0x1F;
    const unsigned int g6 = (p >> 5)  & 0x3F;
    const unsigned int b5 = p & 0x1F;

    const unsigned int r = (r5 << 3) | (r5 >> 2);
    const unsigned int g = (g6 << 2) | (g6 >> 4);
    const unsigned int b = (b5 << 3) | (b5 >> 2);
    const unsigned int gray = (77 * r + 150 * g + 29 * b) >> 8;
    return static_cast<unsigned char>(gray);
}

static unsigned short rgb565(unsigned char r, unsigned char g, unsigned char b) {
    const unsigned short R = static_cast<unsigned short>(r >> 3);
    const unsigned short G = static_cast<unsigned short>(g >> 2);
    const unsigned short B = static_cast<unsigned short>(b >> 3);
    return static_cast<unsigned short>((R << 11) | (G << 5) | B);
}

static void put_pixel_rgb565_stride(volatile unsigned short* fb,
                                    int vis_w,
                                    int vis_h,
                                    int stride_w,
                                    int x,
                                    int y,
                                    unsigned short color) {
    if (x < 0 || y < 0 || x >= vis_w || y >= vis_h) return;
    fb[static_cast<size_t>(y) * stride_w + x] = color;
}

static void draw_roi_box_rgb565_stride(volatile unsigned short* fb,
                                       int vis_w,
                                       int vis_h,
                                       int stride_w,
                                       int x0,
                                       int y0,
                                       int roi_w,
                                       int roi_h,
                                       int thickness,
                                       unsigned short color) {
    const int x1 = x0 + roi_w - 1;
    const int y1 = y0 + roi_h - 1;
    if (thickness < 1) thickness = 1;

    for (int t = 0; t < thickness; ++t) {
        const int yt = y0 + t;
        const int yb = y1 - t;
        for (int x = x0; x <= x1; ++x) {
            put_pixel_rgb565_stride(fb, vis_w, vis_h, stride_w, x, yt, color);
            put_pixel_rgb565_stride(fb, vis_w, vis_h, stride_w, x, yb, color);
        }
    }

    for (int t = 0; t < thickness; ++t) {
        const int xl = x0 + t;
        const int xr = x1 - t;
        for (int y = y0; y <= y1; ++y) {
            put_pixel_rgb565_stride(fb, vis_w, vis_h, stride_w, xl, y, color);
            put_pixel_rgb565_stride(fb, vis_w, vis_h, stride_w, xr, y, color);
        }
    }
}

static void crop_rgb565(const volatile unsigned short* src,
                        int src_stride,
                        int x0,
                        int y0,
                        int roi_w,
                        int roi_h,
                        unsigned short* dst) {
    for (int yy = 0; yy < roi_h; ++yy) {
        const volatile unsigned short* row = src + (y0 + yy) * src_stride + x0;
        unsigned short* out = dst + static_cast<size_t>(yy) * roi_w;
        for (int xx = 0; xx < roi_w; ++xx) out[xx] = row[xx];
    }
}

static GrayImage roi_rgb565_to_grayimage(const unsigned short* roi565, int roi_w, int roi_h) {
    GrayImage out(roi_h, roi_w, 0);
    for (int y = 0; y < roi_h; ++y) {
        for (int x = 0; x < roi_w; ++x) {
            out.at(y, x) = rgb565_to_gray(roi565[static_cast<size_t>(y) * roi_w + x]);
        }
    }
    return out;
}

} // namespace

CaptureDevice::CaptureDevice()
    : fd(-1),
      virtual_base_regs(nullptr),
      virtual_base_image(nullptr),
      video_in_dma(nullptr),
      key_ptr(nullptr),
      ledr_ptr(nullptr),
      video_mem(nullptr),
      roi_x(0),
      roi_y(0),
      roi_w(160),
      roi_h(160),
      border_inset(6),
      initialized(false) {}

bool init_capture_device(CaptureDevice& dev, int roi_w, int roi_h, int border_inset) {
    dev.roi_w = roi_w;
    dev.roi_h = roi_h;
    dev.border_inset = border_inset;
    dev.roi_x = 80;
    dev.roi_y = 0;

    dev.fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (dev.fd == -1) {
        return false;
    }

    dev.virtual_base_regs = mmap(nullptr, HW_REGS_SPAN, PROT_READ | PROT_WRITE,
                                 MAP_SHARED, dev.fd, static_cast<off_t>(HW_REGS_BASE));
    if (dev.virtual_base_regs == MAP_FAILED) {
        dev.virtual_base_regs = nullptr;
        close(dev.fd);
        dev.fd = -1;
        return false;
    }

    dev.virtual_base_image = mmap(nullptr, IMAGE_SPAN, PROT_READ | PROT_WRITE,
                                  MAP_SHARED, dev.fd, static_cast<off_t>(FPGA_ONCHIP_BASE));
    if (dev.virtual_base_image == MAP_FAILED) {
        munmap(dev.virtual_base_regs, HW_REGS_SPAN);
        dev.virtual_base_regs = nullptr;
        dev.virtual_base_image = nullptr;
        close(dev.fd);
        dev.fd = -1;
        return false;
    }

    auto regs_base = reinterpret_cast<std::uintptr_t>(dev.virtual_base_regs);
    dev.video_in_dma = reinterpret_cast<volatile unsigned int*>(regs_base + (VIDEO_BASE & HW_REGS_MASK));
    dev.key_ptr = reinterpret_cast<volatile unsigned short*>(regs_base + (PUSH_BASE & HW_REGS_MASK));
    dev.ledr_ptr = reinterpret_cast<volatile unsigned int*>(regs_base + (LEDR_BASE & HW_REGS_MASK));
    dev.video_mem = reinterpret_cast<volatile unsigned short*>(dev.virtual_base_image);
    dev.initialized = true;
    return true;
}

void shutdown_capture_device(CaptureDevice& dev) {
    if (dev.virtual_base_image) {
        munmap(dev.virtual_base_image, IMAGE_SPAN);
        dev.virtual_base_image = nullptr;
    }
    if (dev.virtual_base_regs) {
        munmap(dev.virtual_base_regs, HW_REGS_SPAN);
        dev.virtual_base_regs = nullptr;
    }
    if (dev.fd != -1) {
        close(dev.fd);
        dev.fd = -1;
    }

    dev.video_in_dma = nullptr;
    dev.key_ptr = nullptr;
    dev.ledr_ptr = nullptr;
    dev.video_mem = nullptr;
    dev.initialized = false;
}

GrayImage capture_roi_gray_on_button(CaptureDevice& dev, bool save_debug_bmp, const std::string& debug_prefix) {
    if (!dev.initialized || !dev.video_in_dma || !dev.key_ptr || !dev.video_mem) {
        throw std::runtime_error("Capture device not initialized");
    }

    const unsigned short box_color = rgb565(255, 0, 0);
    *(dev.video_in_dma + 3) = 0x4;

    while (true) {
        draw_roi_box_rgb565_stride(dev.video_mem, CAP_W, CAP_H, STRIDE_W,
                                   dev.roi_x, dev.roi_y, dev.roi_w, dev.roi_h, 3, box_color);
        if (*dev.key_ptr != 7) {
            break;
        }
        usleep(1000);
    }

    *(dev.video_in_dma + 3) = 0x0;

    int roi_x2 = dev.roi_x + dev.border_inset;
    int roi_y2 = dev.roi_y + dev.border_inset;
    int roi_w2 = dev.roi_w - 2 * dev.border_inset;
    int roi_h2 = dev.roi_h - 2 * dev.border_inset;
    if (roi_w2 < 8) roi_w2 = 8;
    if (roi_h2 < 8) roi_h2 = 8;

    std::vector<unsigned short> roi_buf(static_cast<size_t>(roi_w2) * roi_h2);
    crop_rgb565(dev.video_mem, STRIDE_W, roi_x2, roi_y2, roi_w2, roi_h2, roi_buf.data());

    if (save_debug_bmp) {
        saveImageShort((debug_prefix + "_roi.bmp").c_str(), roi_buf.data(), roi_w2, roi_h2);
    }

    GrayImage gray = roi_rgb565_to_grayimage(roi_buf.data(), roi_w2, roi_h2);
    return gray;
}

void display_digit_ledr(CaptureDevice& dev, int digit) {
    if (!dev.initialized || dev.ledr_ptr == nullptr) {
        return;
    }

    if (digit < 0 || digit > 9) {
        *dev.ledr_ptr = 0x0;
        return;
    }

    // Display digit using LEDR4, LEDR5, LEDR6
    *dev.ledr_ptr = (static_cast<unsigned int>(digit) & 0x7) << 4;
}