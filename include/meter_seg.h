#ifndef _METER_SEG_H_
#define _METER_SEG_H_

#include "net.h"
#include "mat.h"
#include "meter_detect.h"
#include <opencv2/opencv.hpp>

#define DEEPLABV3P_TARGET_SIZE 416  // target image size after resize, might use 416 for small model

class MeterSegmentation {

public:
    MeterSegmentation(const char* param, const char* bin);
    ~MeterSegmentation();
    bool run(const cv::Mat& img, ncnn::Mat& res);
    std::vector<cv::Mat> preprocess(const std::vector<cv::Mat>& input_images, std::vector<cv::Mat>& resize_images);
    std::vector<cv::Mat> cut_roi_img(const cv::Mat& bgr, const std::vector<Object>& objects);
    float LetterBoxImage(const cv::Mat& image, cv::Mat& out_image, const cv::Size& new_shape, const cv::Scalar& color);

private:
    ncnn::Net meterSeg;
    ncnn::Mat input, res;

    const float mean[3] = { 0.485f * 255.f, 0.456f * 255.f, 0.406f * 255.f };
    const float std[3] = { 1 / 0.229f / 255.f, 1 / 0.224f / 255.f, 1 / 0.225f / 255.f };
};

#endif
