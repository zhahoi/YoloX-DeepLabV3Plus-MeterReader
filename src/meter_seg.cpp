#include "meter_seg.h"
#include "gpu.h"
#include <iostream>
 
// 为了防止重复创建
std::vector<cv::Mat> cut_images;
cv::Mat resize_image;
cv::Mat cut_image;
std::vector<cv::Mat> outputs;

MeterSegmentation::MeterSegmentation(const char* param, const char* bin)
{
    meterSeg.opt.use_vulkan_compute = false;
    meterSeg.opt.use_bf16_storage = false;
    meterSeg.load_param(param);
    meterSeg.load_model(bin);
}

float MeterSegmentation::LetterBoxImage(const cv::Mat& image, cv::Mat& out_image, const cv::Size& new_shape, const cv::Scalar& color)
{
    cv::Size shape = image.size();
    float r = std::min((float)new_shape.height / (float)shape.height, (float)new_shape.width / (float)shape.width);

    int newUnpad[2]{ (int)std::round((float)shape.width * r), (int)std::round((float)shape.height * r) };

    cv::Mat tmp;
    if (shape.width != newUnpad[0] || shape.height != newUnpad[1]) {
        cv::resize(image, tmp, cv::Size(newUnpad[0], newUnpad[1]));
    }
    else {
        tmp = image.clone();
    }

    float dw = new_shape.width - newUnpad[0];
    float dh = new_shape.height - newUnpad[1];

    dw /= 2.0f;
    dh /= 2.0f;

    int top = int(std::round(dh - 0.1f));
    int bottom = int(std::round(dh + 0.1f));
    int left = int(std::round(dw - 0.1f));
    int right = int(std::round(dw + 0.1f));

    cv::copyMakeBorder(tmp, out_image, top, bottom, left, right, cv::BORDER_CONSTANT, color);

    return 1.0f / r;
}

float MeterSegmentation::ResizeImage(const cv::Mat& image, cv::Mat& out_image, const cv::Size& new_shape, const cv::Scalar& color = cv::Scalar(128, 128, 128))
{
    cv::Size shape = image.size();
    float scale = std::min(static_cast<float>(new_shape.height) / shape.height, static_cast<float>(new_shape.width) / shape.width);

    int new_width = static_cast<int>(shape.width * scale);
    int new_height = static_cast<int>(shape.height * scale);

    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(new_width, new_height), 0, 0, cv::INTER_CUBIC);

    out_image = cv::Mat(new_shape, image.type(), color);
    resized_image.copyTo(out_image(cv::Rect((new_shape.width - new_width) / 2, (new_shape.height - new_height) / 2, new_width, new_height)));

    return scale;
}

void MeterSegmentation::Softmax(ncnn::Mat& res)
{
    for (int i = 0; i < res.h; i++)
    {
        for (int j = 0; j < res.w; j++)
        {
            float max = -FLT_MAX;
            for (int q = 0; q < res.c; q++)
            {
                max = std::max(max, res.channel(q).row(i)[j]);
            }

            float sum = 0.0f;
            for (int q = 0; q < res.c; q++)
            {
                res.channel(q).row(i)[j] = exp(res.channel(q).row(i)[j] - max);
                sum += res.channel(q).row(i)[j];
            }

            for (int q = 0; q < res.c; q++)
            {
                res.channel(q).row(i)[j] /= sum;
            }
        }
    }
}

bool MeterSegmentation::run(const cv::Mat& img, ncnn::Mat& res)
{
    if (img.empty())
        return false;

    // 进行预处理
    ncnn::Mat input = ncnn::Mat::from_pixels(img.data, ncnn::Mat::PIXEL_BGR2RGB, DEEPLABV3P_TARGET_SIZE, DEEPLABV3P_TARGET_SIZE);
    input.substract_mean_normalize(mean, std);

    ncnn::Extractor ex = meterSeg.create_extractor();
    ex.input("images", input);
    ex.extract("output", res);

    // softmax
    Softmax(res);

    return true;
}

MeterSegmentation::~MeterSegmentation()
{
    meterSeg.clear();
}

std::vector<cv::Mat> MeterSegmentation::cut_roi_img(const cv::Mat& bgr, const std::vector<Object>& objects)
{
    cut_images.clear();
    cv::Mat image = bgr.clone();

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        fprintf_s(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
            obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        cut_image = image(obj.rect);
        cut_images.push_back(cut_image);

#ifdef VISUALIZE
        cv::imshow("sub_image", cut_image);
        cv::waitKey(0);
#endif // VISUALIZE
    }
    return cut_images;
}

std::vector<cv::Mat> MeterSegmentation::preprocess(const std::vector<cv::Mat>& input_images, std::vector<cv::Mat>& resize_images)
{
    int meter_num = input_images.size();
    outputs.clear();

    for (int i_num = 0; i_num < meter_num; i_num++)
    {
        cv::Mat input_image = input_images[i_num].clone();

        std::cout << "current scale: " << input_image.rows << ", " << input_image.cols << std::endl;

#ifdef VISUALIZE
        cv::imshow("input_image: ", input_image);
        cv::waitKey(0);
#endif
       
        // 直接进行resize
        // cv::resize(input_image, resize_image, cv::Size(DEEPLABV3P_TARGET_SIZE, DEEPLABV3P_TARGET_SIZE), 0, 0, cv::INTER_AREA);
        
        // 对输入图像进行不失真的resize,加灰条
        // float scale = LetterBoxImage(input_image, resize_image, cv::Size(DEEPLABV3P_TARGET_SIZE, DEEPLABV3P_TARGET_SIZE), cv::Scalar(128, 128, 128));
        float scale = ResizeImage(input_image, resize_image, cv::Size(DEEPLABV3P_TARGET_SIZE, DEEPLABV3P_TARGET_SIZE), cv::Scalar(128, 128, 128));
        std::cout << "current scale: " << scale << std::endl;

#ifdef VISUALIZE
        cv::imshow("resize_image", resize_image);
        cv::waitKey(0);
#endif

        ncnn::Mat res;
        run(resize_image, res);

#ifdef VISUALIZE
        std::cout << "Res shape: " << res.h << ", " << res.w << ", " << res.c << std::endl;

#endif
        cv::Mat mask(DEEPLABV3P_TARGET_SIZE, DEEPLABV3P_TARGET_SIZE, CV_8UC1, cv::Scalar(0, 0, 0));
  
        const float* class0mask = res.channel(0); // background
        const float* class1mask = res.channel(1); // pointer
        const float* class2mask = res.channel(2); // scale

        // 输出一种掩码图，让背景数值为0，指针数值为1，刻度数值为2
        for (int i = 0; i < DEEPLABV3P_TARGET_SIZE; i++)
        {
            for (int j = 0; j < DEEPLABV3P_TARGET_SIZE; j++)
            {
                int num = i * DEEPLABV3P_TARGET_SIZE + j;
                if ((class1mask[num] > class2mask[num]) && (class1mask[num] > class0mask[num]))
                {
                    mask.at<uchar>(i, j) = 1;
                }
            
                if ((class2mask[num] > class1mask[num]) && (class2mask[num] > class0mask[num]))
                {
                    mask.at<uchar>(i, j) = 2;
                }
            }
        }
      
        
#ifdef VISUALIZE
        std::cout << "Res shape: " << res.h << ", " << res.w << ", " << res.c << std::endl;
#endif
       
        outputs.push_back(mask);
        resize_images.push_back(resize_image);

        mask.release();
        input_image.release();
        resize_image.release();
        res.release();
    }

    return outputs;
}
