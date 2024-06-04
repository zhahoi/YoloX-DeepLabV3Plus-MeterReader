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
    meterSeg.opt.use_bf16_storage = true;
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

bool MeterSegmentation::process(cv::Mat& image, ncnn::Mat& input)
{
    if (image.empty())
        return false;
    int img_w = image.cols;
    int img_h = image.rows;
   
    cv::Mat float_Mat;
    // image.convertTo(float_Mat, CV_32F, 1.0/255, 0);
    input = ncnn::Mat::from_pixels(image.data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h);
    input.substract_mean_normalize(mean, std);
    const float* pInput = input.channel(0);

    return true;
}

bool MeterSegmentation::run(cv::Mat& img, ncnn::Mat& res)
{
    ncnn::Mat input;
    bool is_process = process(img, input);
    const float* pInput = input.channel(0);
    if (!is_process)
    {
        std::cout << "The image is invlid!\n";
        return false;
    }
    ncnn::Extractor ex = meterSeg.create_extractor();
    ex.input("images", input);
    ex.extract("output", res);

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
        // 这里发现如果进行不失真的resize结果可能会不准确，所以直接粗暴地进行resze了，原因可能是训练模型的时候就没有加灰条
        cv::resize(input_image, resize_image, cv::Size(DEEPLABV3P_TARGET_SIZE, DEEPLABV3P_TARGET_SIZE), 0, 0, cv::INTER_LINEAR);
        /*
        // 需要对输入图像进行不失真的resize
        float scale = LetterBoxImage(input_image, resize_image, cv::Size(DEEPLABV3P_TARGET_SIZE, DEEPLABV3P_TARGET_SIZE), cv::Scalar(114, 114, 114));

        std::cout << "current scale: " << scale << std::endl;
        */
        // cv::imshow("resize_image", resize_image);
        // cv::waitKey(0);

        ncnn::Mat res;
        run(resize_image, res);

#ifdef VISUALIZE
        std::cout << "Res shape: " << res.h << ", " << res.w << ", " << res.c << std::endl;

#endif
        cv::Mat mask(DEEPLABV3P_TARGET_SIZE, DEEPLABV3P_TARGET_SIZE, CV_8UC1);
        uchar* pMask = mask.data;

        const float* class0mask = res.channel(0); // background
        const float* class1mask = res.channel(1); // pointer
        const float* class2mask = res.channel(2); // scale

        // 输出一种掩码图，让背景数值为0，指针数值为1，刻度数值为2
        for (int i = 0; i < DEEPLABV3P_TARGET_SIZE * DEEPLABV3P_TARGET_SIZE; i++)
        {
            // 0: background 1：pointer 2: scale
            // pMask[i] = class0mask[i] > class1mask[i] ? 0 : 1
            if (class0mask[i] > class1mask[i])
            {
                if (class0mask[i] > class2mask[i])
                {
                    pMask[i] = 0;  // 背景
                }
                else
                {
                    pMask[i] = 2;  // 刻度
                }
            }
            else
            {
                if (class1mask[i] > class2mask[i])
                {
                    pMask[i] = 1;  // 指针
                }
                else
                {
                    pMask[i] = 2;  // 刻度
                }
            }
        }
        
#ifdef VISUALIZE
        std::cout << "Res shape: " << res.h << ", " << res.w << ", " << res.c << std::endl;
#endif

        /*
        // 使用scale判断当前的图片大小是否大于resize后的图像，然后再进行返回
        cv::Rect roi_rect;
        cv::Mat mask_resize, mask_recover;
        int x_offset, y_offset;

        int max_size = (input_image.rows > input_image.cols) ? input_image.rows : input_image.cols;

        std::cout << "max_size: " << max_size << std::endl;

        if (scale >= 1)  // 说明检测图像大于resize后的图像
        {
            x_offset = (max_size - input_image.cols) / 2;
            y_offset = (max_size - input_image.rows) / 2;

            std::cout << "current offset: " << x_offset << ", " << y_offset << std::endl;
            cv::resize(mask, mask_resize, cv::Size(max_size, max_size));

            if (input_image.rows >= input_image.cols)
                roi_rect.x = x_offset;
                roi_rect.y = y_offset;
                roi_rect.width = input_image.cols;
                roi_rect.height = input_image.rows;

            std::cout << "rect: " << roi_rect.x << ", " << roi_rect.y << ", " << roi_rect.width << ", " << roi_rect.height << std::endl;

            mask_recover = mask_resize(roi_rect);
        }
        else // 说明检测图像小于resize后的图像
        {
            x_offset = (DEEPLABV3P_TARGET_SIZE * scale - input_image.cols) / 2;
            y_offset = (DEEPLABV3P_TARGET_SIZE * scale - input_image.rows) / 2;

            std::cout << "current offset: " << x_offset << ", " << y_offset << std::endl;

            roi_rect.x = x_offset;
            roi_rect.y = y_offset;
            roi_rect.width = DEEPLABV3P_TARGET_SIZE * scale - x_offset;
            roi_rect.height = DEEPLABV3P_TARGET_SIZE * scale - y_offset;

            mask_recover = mask(roi_rect);
        }

        // cv::imshow("mask_recover", mask_recover);
        // cv::waitKey(0);
         */
        outputs.push_back(mask);
        resize_images.push_back(resize_image);

        // outputs.push_back(mask_recover);
    }

    return outputs;
}
