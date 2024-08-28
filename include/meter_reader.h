#ifndef METER_READER_H
#define METER_READER_H

#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include "meter_detect.h"
#include "meter_seg.h"

#define SCALE_BEGINNING 0 // -0.1
#define SCALE_END  0.1

class MeterReader {
public:
    void drawMask(cv::Mat& img, cv::Mat& mask, float scale_value);
    float reader_process(cv::Mat& img, cv::Mat& mask);
    std::vector<float> multi_read_process(const std::vector<cv::Mat>& input_image, const std::vector<cv::Mat>& output);
    cv::Mat result_visualizer(const cv::Mat& bgr, const std::vector<Object>& objects_remains, const std::vector<float> scale_values);

private:
    std::string floatToString(const float& val);
    std::vector<int> argsort(const std::vector<int>& array);
    double getDistance(cv::Point2f point1, cv::Point2f point2);
    int thresholdByCategory(cv::Mat& src, cv::Mat& dst, int category);
    int thresholdByContour(cv::Mat& src, cv::Mat& dst, std::vector<cv::Point2f>& contour);
    int getScaleLocation(cv::Mat& dial_mask, cv::Point2f* locations);
    int getCenterLocation(cv::Mat& dial_mask, cv::Point2f& center_location);  // 求解表盘的中心点
    // int getCenterLocation(cv::Mat& img, cv::Point2f& center_location);  // 求解表盘的中心点
    int getMinAreaRectPoints(cv::Mat& pointer_mask, cv::Point2f* Ps);
    int getPointerVertexIndex(cv::Point2f& center_location, cv::Point2f* Ps, int& vertex_index);
    int getPointerLocation(cv::Mat& pointer_mask, cv::Point2f& center_location, cv::Point2f* pointer_location);
    float getAngleRatio(cv::Point2f* scale_locations, cv::Point2f& pointer_head_location, cv::Point2f& center_location);
    float getScaleValue(float angleRatio);
    void vis_for_test(cv::Mat& img, cv::Point2f* scale_locations, cv::Point2f* pointer_locations,
        cv::Point2f& center_location, float scale_value);

    cv::Mat gray_img, gray_cut_img, gray_open_img, bin_img, canny_img, show_img;
    cv::Point center;
    int center_radius;
};


#endif //METER_READER_H
