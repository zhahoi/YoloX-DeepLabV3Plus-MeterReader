#include "meter_seg.h"
#include "meter_reader.h"
#include "meter_detect.h"

#include <float.h>
#include <stdio.h>
#include <vector>

#define det_param  "C:/CPlusPlus/MetersReader/models/yolox_s.param"
#define det_bin "C:/CPlusPlus/MetersReader/models/yolox_s.bin"
#define seg_param "C:/CPlusPlus/MetersReader/models/deeplabv3p_f16.param"
#define seg_bin "C:/CPlusPlus/MetersReader/models/deeplabv3p_f16.bin"
#define rtsp_dir "rtsp://admin:Admin123@192.168.1.13:554/snl/live/1/1"

MeterDetection* meterDet = new MeterDetection(det_param, det_bin);
MeterSegmentation* meterSeg = new MeterSegmentation(seg_param, seg_bin);
MeterReader meterReader;

std::vector<cv::Mat> meters_image;
std::vector<cv::Mat> meters_image_pad;
std::vector<Object> objects;
std::vector<cv::Mat> seg_result;
std::vector<float> scale_values;

int process(cv::Mat& input_image)
{
	cv::Mat m = input_image.clone();
	if (m.empty())
	{
		printf("cv::imread read image failed\n");
		return -1;
	}

	objects.clear();
	meterDet->detect_objects(m, objects);

	std::cout << "object size: " << objects.size() << std::endl;
	// ::Mat img_vis = meterDet->draw_objects(m, objects);

	// 如果检测到目标
	if (objects.size() > 0)
	{
		// meters_image中存放的是仪表图片
		meters_image.clear();
		meters_image = meterSeg->cut_roi_img(m, objects);

		// 对仪表图片进行分割操作
		seg_result.clear();
		meters_image_pad.clear();
		seg_result = meterSeg->preprocess(meters_image, meters_image_pad);

		// 用来存放检测结果
		scale_values.clear();
		scale_values = meterReader.multi_read_process(meters_image_pad, seg_result);

		// 对检测结果进行可视化
		cv::Mat img_vis = meterReader.result_visualizer(m, objects, scale_values);

		cv::imshow("Video", img_vis);
	}
	else 
	{
		cv::imshow("Video", m);
	}

	cv::waitKey(1);
	
	return 0;
}


int main(int argc, char** argv)
{
	/*
	if (argc != 2)
	{
		fprintf(stderr, "Usage: %s [image] or [video] \n", argv[0]);
		return -1;
	}
	
	std::string detect_method = argv[1];
	
	if (detect_method == "image")
	{
	
		const char* input_image_path = "C:/CPlusPlus/MetersReader/models/50.jpg";
		cv::Mat input_image = cv::imread(input_image_path);
		process(input_image);
	}
	else if (detect_method == "video")
	{
	*/
	// int devicenum = 0;
	cv::VideoCapture cap; // 打开摄像头函数

	// 读取rtsp
	cap.open(rtsp_dir);
	// cv::VideoCapture cap(devicenum);
	cap.set(CV_CAP_PROP_FRAME_WIDTH, 1280);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, 720);

	if (!cap.isOpened())
	{
		printf("open camera failed.\n");
		return -1;
	}
	printf("open camera succ\n");

	while (1)
	{
		cv::Mat input_image;
		cap >> input_image;
		printf("-------------------------\n");
		printf("got camera image.\n");

		process(input_image);
	}
	cap.release();
	cv::destroyAllWindows();
	// }

	printf("meter end\n");
	return 0;
}