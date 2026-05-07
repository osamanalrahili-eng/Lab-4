#ifndef CAPTURE_DIGIT_H
#define CAPTURE_DIGIT_H

#include <string>
#include "preprocess_digit.h"

struct CaptureDevice {
    int fd;
    void* virtual_base_regs;
    void* virtual_base_image;
    volatile unsigned int* video_in_dma;
    volatile unsigned short* key_ptr;
    volatile unsigned int* ledr_ptr;
    volatile unsigned short* video_mem;

    int roi_x;
    int roi_y;
    int roi_w;
    int roi_h;
    int border_inset;
    bool initialized;

    CaptureDevice();
};

bool init_capture_device(
    CaptureDevice& dev,
    int roi_w = 160,
    int roi_h = 160,
    int border_inset = 6
);

void shutdown_capture_device(CaptureDevice& dev);

GrayImage capture_roi_gray_on_button(
    CaptureDevice& dev,
    bool save_debug_bmp = false,
    const std::string& debug_prefix = "capture"
);
void display_digit_ledr(CaptureDevice& dev, int digit);
#endif
