#ifndef _METER_READERV2_H_
#define _METER_READERV2_H_	 

#include "meter_detect.h"
#include <vector>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <limits>
#include <cmath>
#include <cstdio>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>

#define METER_TYPE_NUM 3
#define TYPE_THRESHOLD_HIGH 40
#define TYPE_THRESHOLD_LOW 25

struct READ_RESULT {
	int   scale_num;
	float scales;
	float ratio;
};

typedef struct MeterConfig {
	float scale_value;
	float range;
	char  str[10];
}MeterConfig_T;

extern MeterConfig_T meter_config[];

class MeterReaderV2 {
public:
	// void read_process(std::vector<std::vector<unsigned char>>& seg_image, std::vector<READ_RESULT>& read_results)
	void read_process(const std::vector<cv::Mat>& seg_image, std::vector<READ_RESULT>& read_results);
	std::vector<float> get_result(const std::vector<READ_RESULT>& read_results);

private:
	void scale_mean_filtration(std::vector<unsigned int>& scale_data, std::vector<unsigned int>& scale_mean_data);
	READ_RESULT get_meter_reader(std::vector<unsigned int>& scale, std::vector<unsigned int>& pointer);
	std::vector<unsigned char> mat2vector(const cv::Mat& mat);
	void vector2mat(std::vector<unsigned int>& a, const char* str);
	void creat_line_image(std::vector<unsigned char>& seg_image, std::vector<unsigned char>& output);
	void convert_1D_data(std::vector<unsigned char>& line_image, std::vector<unsigned int>& scale_data, std::vector<unsigned int>& pointer_data);
};

#endif
