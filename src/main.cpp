#include "meter_seg.h"
#include "meter_reader.h"
#include "meter_detect.h"

#include <float.h>
#include <stdio.h>
#include <vector>

// 用于多线程
#include <iostream>
#include <chrono>
#include <thread>
#include <mutex>
#include <atomic>
#include <queue>
#include <conio.h>
#include <windows.h>
#include <time.h>

#define det_param  "C:/CPlusPlus/MetersReader/models/yolox_s.param"
#define det_bin "C:/CPlusPlus/MetersReader/models/yolox_s.bin"
#define seg_param "C:/CPlusPlus/MetersReader/models/deeplabv3p_f16.param"
#define seg_bin "C:/CPlusPlus/MetersReader/models/deeplabv3p_f16.bin"
#define rtsp_dir "rtsp://admin:Admin123@192.168.1.13:554/snl/live/1/1"
#define devicenum 0
#define video_width 1280
#define video_height 720

// #define USE_RTSP 1

static std::mutex mutex;  // 进程锁
static std::atomic_bool isOpen;
static std::queue<cv::Mat> frames;  // 先进先出队列
static bool is_image; 

MeterDetection* meterDet = new MeterDetection(det_param, det_bin);
MeterSegmentation* meterSeg = new MeterSegmentation(seg_param, seg_bin);
MeterReader meterReader;

std::vector<cv::Mat> meters_image;
std::vector<cv::Mat> meters_image_pad;
std::vector<Object> objects;
std::vector<cv::Mat> seg_result;
std::vector<float> scale_values;
cv::Mat frame, img_vis;


void process(const cv::Mat& input_image)
{
	if (input_image.empty())
	{
		printf("cv::imread read image failed\n");
		return;
	}

	objects.clear();
	meterDet->detect_objects(input_image, objects);

	std::cout << "object size: " << objects.size() << std::endl;
	// ::Mat img_vis = meterDet->draw_objects(m, objects);

	// 如果检测到目标
	if (objects.size() > 0)
	{
		// meters_image中存放的是仪表图片
		meters_image.clear();
		meters_image = meterSeg->cut_roi_img(input_image, objects);

		// 对仪表图片进行分割操作
		seg_result.clear();
		meters_image_pad.clear();
		seg_result = meterSeg->preprocess(meters_image, meters_image_pad);

		// 用来存放检测结果
		scale_values.clear();
		scale_values = meterReader.multi_read_process(meters_image_pad, seg_result);

		// 对检测结果进行可视化
		img_vis = meterReader.result_visualizer(input_image, objects, scale_values);

		if (img_vis.empty()) {
			return;
		}
		cv::imshow("frame", img_vis);
	}
	else
	{
		cv::imshow("frame", input_image);
	}

	if (!is_image)
	{
		cv::waitKey(1);
		if (cv::waitKey(1) == 'q') {
			return;
		}
	}
	else
	{
		cv::waitKey(0);
	}
}


static void getCameraFrame(std::string rtspDir, int deviceNum, int height, int width)
{
	std::cout << "open camera ..." << std::endl;
#ifdef USE_RTSP
	// 读取rtsp
	cv::VideoCapture cap; // 打开摄像头函数
	cap.open(rtspDir);
#else
	// 电脑摄像头读取
	cv::VideoCapture cap(devicenum);
#endif // USE_RTSP

	cap.set(CV_CAP_PROP_FRAME_WIDTH, width);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, height);
	cap.set(CV_CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));  // 视频流格式
	cap.set(CV_CAP_PROP_FPS, 30);  // 帧率  帧/秒

	if (!cap.isOpened())
	{
		isOpen = false;
		std::cout << "Failed to open camera with rtsp dir " << rtspDir << std::endl;
		return;
	}
	printf("open camera succeed\n");

	while (isOpen)
	{
		cap >> frame;
		mutex.lock();  // 修改公共资源，用进程锁锁定
		frames.push(frame);
		mutex.unlock();
	}  // 被主进程通知结束循环
	std::cout << "releasing camera occupied..." << std::endl;
	cap.release();
	std::cout << "release camera occupied succeed..." << std::endl;
}


static void processCameraFrame()
{
	cv::namedWindow("frame");  // 创建显示窗口
	while (isOpen)
	{
		if (!frames.empty())  // 队列里有帧
		{
			mutex.lock();
			frame = frames.front();
			frames.pop();  // 帧出队
			mutex.unlock();

			auto start = std::chrono::high_resolution_clock::now();  // 记录开始时间
			process(frame);
			auto end = std::chrono::high_resolution_clock::now();  // 记录开始时间

			std::chrono::duration<double, std::milli> duration = end - start;   // 计算执行时间
			std::cout << "execute time： " << duration.count() << " ms" << std::endl;
		}
	}
	cv::destroyWindow("frame");
	isOpen = false;
}


int main(int argc, char** argv)
{
	if (argc != 2)
	{
		fprintf(stderr, "Usage: %s [image] or [video] \n", argv[0]);
		return -1;
	}

	std::string detect_method = argv[1];
	is_image = false; // 初设

	if (detect_method == "image")
	{
		is_image = true; 
		const char* input_image_path = "C:/CPlusPlus/MetersReader/models/new_meter.jpg";
		cv::Mat input_image = cv::imread(input_image_path);
		process(input_image);
	}
	else if (detect_method == "video")
	{

		std::thread t1(getCameraFrame, rtsp_dir, devicenum, video_height, video_width);
		t1.detach();

		std::thread t2(processCameraFrame);
		t2.detach();

		isOpen = true;  // 初设

		while (1)
		{
			Sleep(1);
			if (_kbhit())
				break;
		}
		isOpen = false;
		Sleep(1);  // 等待进程结束
	}

	printf("meter end\n");
	return 0;
}
