#ifndef METER_DETECT_H
#define METER_DETECT_H

#include "layer.h"
#include "net.h"

#if defined(USE_NCNN_SIMPLEOCV)
#include "simpleocv.h"
#else
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif
#include <float.h>
#include <stdio.h>
#include <vector>

#define YOLOX_NMS_THRESH  0.45 // nms threshold
#define YOLOX_CONF_THRESH 0.75 // threshold of bounding box prob
#define YOLOX_TARGET_SIZE 640  // target image size after resize, might use 416 for small model

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

struct GridAndStride
{
    int grid0;
    int grid1;
    int stride;
};


class MeterDetection {

public:
    MeterDetection(const char* param, const char* bin);
    ~MeterDetection();
    // function
    inline float intersection_area(const Object& a, const Object& b);
    void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right);
    void qsort_descent_inplace(std::vector<Object>& objects);
    void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold);
    void generate_grids_and_stride(const int target_w, const int target_h, std::vector<int>& strides, std::vector<GridAndStride>& grid_strides);
    void generate_yolox_proposals(std::vector<GridAndStride> grid_strides, const ncnn::Mat& feat_blob, float prob_threshold, std::vector<Object>& objects);
    cv::Mat draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects);
    bool detect_objects(cv::Mat& image, std::vector<Object>& objects);

private:
    ncnn::Net yolox;
};

#endif