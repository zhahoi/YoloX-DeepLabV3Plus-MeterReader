#include "meter_seg.h"
#include "meter_reader.h"
#include "meter_detect.h"
#include "meter_readerv2.h"

#include <float.h>
#include <stdio.h>
#include <vector>

// ���ڶ��߳�
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
#define seg_param "C:/CPlusPlus/MetersReader/models/deeplabv3p_fp16.param"
#define seg_bin "C:/CPlusPlus/MetersReader/models/deeplabv3p_fp16.bin"
#define rtsp_dir "rtsp://admin:Admin123@192.168.1.13:554/snl/live/1/1"
#define devicenum 0
#define video_width 1280
#define video_height 720

// #define USE_RTSP 1

static std::mutex mutex;  // ������
static std::atomic_bool isOpen;
static std::queue<cv::Mat> frames;  // �Ƚ��ȳ�����
static bool is_image; 

MeterDetection* meterDet = new MeterDetection(det_param, det_bin);
MeterSegmentation* meterSeg = new MeterSegmentation(seg_param, seg_bin);
MeterReader meterReader;
MeterReaderV2 meterReaderV2;

std::vector<cv::Mat> meters_image;
std::vector<cv::Mat> meters_image_pad;
std::vector<Object> objects;
std::vector<cv::Mat> seg_result;
std::vector<float> scale_values;
cv::Mat frame, img_vis;
std::vector<READ_RESULT> read_results;

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

	// �����⵽Ŀ��
	if (objects.size() > 0)
	{
		// meters_image�д�ŵ����Ǳ�ͼƬ
		meters_image.clear();
		meters_image = meterSeg->cut_roi_img(input_image, objects);

		// ���Ǳ�ͼƬ���зָ����
		seg_result.clear();
		meters_image_pad.clear();
		seg_result = meterSeg->preprocess(meters_image, meters_image_pad);

		// ��һ�ַ���
		// read_results.clear();
		// meterReaderV2.read_process(seg_result, read_results);

		// ������ż����
		scale_values.clear();
		scale_values = meterReader.multi_read_process(meters_image_pad, seg_result);
		// scale_values = meterReaderV2.get_result(read_results);

		// �Լ�������п��ӻ�
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
	// ��ȡrtsp
	cv::VideoCapture cap; // ������ͷ����
	cap.open(rtspDir);
#else
	// ��������ͷ��ȡ
	cv::VideoCapture cap(devicenum);
#endif // USE_RTSP

	cap.set(CV_CAP_PROP_FRAME_WIDTH, width);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, height);
	cap.set(CV_CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));  // ��Ƶ����ʽ
	cap.set(CV_CAP_PROP_FPS, 30);  // ֡��  ֡/��

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
		mutex.lock();  // �޸Ĺ�����Դ���ý���������
		frames.push(frame);
		mutex.unlock();
	}  // ��������֪ͨ����ѭ��
	std::cout << "releasing camera occupied..." << std::endl;
	cap.release();
	std::cout << "release camera occupied succeed..." << std::endl;
}


static void processCameraFrame()
{
	cv::namedWindow("frame");  // ������ʾ����
	while (isOpen)
	{
		if (!frames.empty())  // ��������֡
		{
			mutex.lock();
			frame = frames.front();
			frames.pop();  // ֡����
			mutex.unlock();

			auto start = std::chrono::high_resolution_clock::now();  // ��¼��ʼʱ��
			process(frame);
			auto end = std::chrono::high_resolution_clock::now();  // ��¼��ʼʱ��

			std::chrono::duration<double, std::milli> duration = end - start;   // ����ִ��ʱ��
			std::cout << "execute time�� " << duration.count() << " ms" << std::endl;
		}
	}
	cv::destroyWindow("frame");
	isOpen = false;
}


int main(int argc, char** argv)
{
	/*
	if (argc != 2)
	{
		fprintf(stderr, "Usage: %s [image] or [video] \n", argv[0]);
		return -1;
	}
	*/
	// std::string detect_method = argv[1];
	std::string detect_method = "video";

	is_image = false; // ����

	if (detect_method == "image")
	{
		is_image = true; 
		// C:/Dataset/MeterVideo/meter_label/images/20190822_23.jpg
		// "C:/CPlusPlus/MetersReader/models/new_meter.jpg";
		const char* input_image_path = "C:/Dataset/MeterVideo/meter_label/images/20190822_1.jpg";
		cv::Mat input_image = cv::imread(input_image_path);
		process(input_image);
	}
	else if (detect_method == "video")
	{

		std::thread t1(getCameraFrame, rtsp_dir, devicenum, video_height, video_width);
		t1.detach();

		std::thread t2(processCameraFrame);
		t2.detach();

		isOpen = true;  // ����

		while (1)
		{
			Sleep(1);
			if (_kbhit())
				break;
		}
		isOpen = false;
		Sleep(1);  // �ȴ����̽���
	}

	printf("meter end\n");
	return 0;
}