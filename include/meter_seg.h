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
    bool run(cv::Mat& img, ncnn::Mat& res);
    std::vector<cv::Mat> preprocess(const std::vector<cv::Mat>& input_images, std::vector<cv::Mat>& resize_images);
    std::vector<cv::Mat> cut_roi_img(const cv::Mat& bgr, const std::vector<Object>& objects);
    float LetterBoxImage(const cv::Mat& image, cv::Mat& out_image, const cv::Size& new_shape, const cv::Scalar& color);

private:
    ncnn::Net meterSeg;
    const float mean[3] = { 0.45734706f * 255.f, 0.43338275f * 255.f, 0.40058118f * 255.f };
    const float std[3] = { 1 / 0.23965294 / 255.f, 1 / 0.23532275 / 255.f, 1 / 0.2398498 / 255.f };
    bool process(cv::Mat& image, ncnn::Mat& input);

};

#endif
