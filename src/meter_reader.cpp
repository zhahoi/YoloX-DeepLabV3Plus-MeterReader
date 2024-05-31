#include "meter_reader.h"
#pragma warning( disable : 4996 )

std::vector<float> scale_values_;
cv::Mat image_, resize_image_, mask_;

std::string MeterReader::floatToString(const float& val)
{
    char* chCode;
    chCode = new char[20];
    sprintf(chCode, "%.2lf", val);
    std::string str(chCode);
    delete[]chCode;
    return str;
}

// ʵ��argsort����
std::vector<int> MeterReader::argsort(const std::vector<int>& array) {

    int size = array.size();
    std::vector<int> array_index(size, 0); // ��ʼ��һ������Ϊsize,ֵȫΪ0������
    for (int i = 0; i < size; i++) {
        array_index[i] = i;
    }
    // sort ������������Lambda���ʽ�� ���ʽ[]���Ǵ���Ĳ���
    std::sort(array_index.begin(), array_index.end(), [&array](int pos1, int pos2){
        return array[pos1] < array[pos2];
    });

    return array_index;
}

// ���������ľ���
double MeterReader::getDistance(cv::Point2f point1, cv::Point2f point2) {

    double distance = sqrtf(powf(point1.x - point2.x, 2) + powf(point1.y - point2.y, 2));
    return distance;
}

// ��ֵ��������������ָ��src����ͨ��ͼ�񣩡����dst(��src��ͬ��״����ָ�����ID
// ���ص�ĻҶ�ֵ�������ָ�����ID������ֵ��Ϊ255��������Ϊ0
// cv::Mat dst = cv::Mat(src.rows, src.cols, CV_8UC1);
int MeterReader::thresholdByCategory(cv::Mat& src, cv::Mat& dst, int category) {
    uchar* p = NULL;
    uchar* q = NULL;
    // ����ָ��ķ�ʽ����ȡCV::Mat
    for (int row = 0; row < src.rows; row++) {
        p = src.ptr<uchar>(row);
        q = dst.ptr<uchar>(row);
        for (int col = 0; col < src.cols; col++) {
            if (p[col] == category) {
                q[col] = 255;
            }
            else {
                q[col] = 0;
            }
        }
    }
    return 0;
}

// ������һ����ֵ����ͼ�����ָ�������������Ϊ0
int MeterReader::thresholdByContour(cv::Mat& src, cv::Mat& dst, std::vector<cv::Point2f>& contour) {
    uchar* p = NULL;
    uchar* q = NULL;
    //    cv::Mat dst = cv::Mat(src.rows, src.cols, CV_8UC1);
    for (size_t row = 0; row < src.rows; row++) {
        p = src.ptr<uchar>(row);
        q = dst.ptr<uchar>(row);
        for (size_t col = 0; col < src.cols; col++) {
            if (pointPolygonTest(contour, cv::Point(col, row), false) == 1) {
                q[col] = 0;
            }
            else {
                q[col] = p[col];
            }
        }
    }
    return 0;
}


// ��ȡ���̵������յ�λ��
// ������Ǳ��̶�ֵ��ͼ��ע�������Ҫ����һ�θ�ʴ���������˵����ǵ����
int MeterReader::getScaleLocation(cv::Mat& dial_mask, cv::Point2f* locations) {
    // ������̵����λ�ú��յ�λ��
    cv::Point2f beginning(-1, -1);
    cv::Point2f end(-1, -1);

    uchar* tmp = NULL;
    for (int row = dial_mask.rows - 1; row >= 0; row--) {
        tmp = dial_mask.ptr<uchar>(row);
        for (int col = 0; col < dial_mask.cols; col++) {
            int value = tmp[col];
            if (value == 255) {
                // �������λ����ͼ�����࣬�����λ��δ��ֵ
                if (col < dial_mask.cols / 2 && beginning.x == -1) {
                    beginning.x = col;
                    beginning.y = row;
                }
                // �����յ�λ����ͼ���Ұ�࣬���յ�λ��δ��ֵ
                if (col >= dial_mask.cols / 2 && end.x == -1) {
                    end.x = col;
                    end.y = row;
                }
            }
        }
        if (beginning.x >= 0 && end.x >= 0) {
            locations[0] = beginning;
            locations[1] = end;
            return 0;
        }
    }
    return -1;
}


// ��ȡ�Ǳ����ĵ�λ��
int MeterReader::getCenterLocation(cv::Mat& img, cv::Point2f& center_location) {
    // get gray image
    cv::cvtColor(img, gray_img, cv::COLOR_BGR2GRAY);

    // median blur
    cv::medianBlur(gray_img, gray_img, 3);
    // cv::GaussianBlur(gray_img, gray_img, cv::Size(3, 3), 0);

    int kernel_size = 3;
    cv::Mat kernel = getStructuringElement(cv::MORPH_RECT, cv::Size(kernel_size, kernel_size));

    // erode and dilate(������)
    cv::morphologyEx(gray_img, gray_open_img, cv::MORPH_OPEN, kernel);

    cv::threshold(gray_open_img, bin_img, 115, 255, cv::THRESH_BINARY);
    cv::medianBlur(bin_img, bin_img, 3);

    // get cutted gray image
    gray_cut_img = gray_img(cv::Range(center.y = gray_img.size().height * 0.38, gray_img.size().height * 0.63),
        cv::Range(center.x = gray_img.size().width * 0.38, gray_img.size().width * 0.63));

    cv::Canny(gray_cut_img, canny_img, 100, 200, 3);

    for (int param2 = 60; param2 >= 20; param2--)
    {
        std::vector<cv::Vec3f> pcircles;
        cv::HoughCircles(canny_img, pcircles, cv::HOUGH_GRADIENT, 1, 10, 80, param2, 2, 200);
        for (auto cc : pcircles)
        {
            center_location.x += cc[0];
            center_location.y += cc[1];
            center_radius = cc[2];

            center_location.x += gray_img.size().height * 0.38;
            center_location.y += gray_img.size().width * 0.38;

            // std::cout << "center_location.x" << center_location.x << ", " << "center_location.y" << center_location.y << std::endl;
          
            /*
            // ��Բ
            cv::circle(img, center_location, center_radius, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
            // ��Բ��
            cv::circle(img, center_location, 2, cv::Scalar(255, 0, 0), 2, cv::LINE_AA);
            
            cv::imshow("output", img);
            cv::waitKey(0);
            */
            return true;
            
        }
        // std::cout << "pre_center_location.x" << center_location.x << ", " << "pre_center_location.y" << center_location.y << std::endl;
    }
    return false;
}


/*
int MeterReader::getCenterLocation(cv::Mat& dial_mask, cv::Point2f& center_location) {

    // ֱ�����ҵ��Ǳ��������Ҳ����ص�λ�ü�ȥ��������ص�λ�õ����ֵ����ֱ����
    int diameter = 0;
    // ����ֱ��������ͨ������һ����������������Ҫ�õ����������ĸ�����
    std::vector<int> diameter_index(4, -1);
    uchar* tmp = NULL;
    for (int row = dial_mask.rows - 1; row >= 0; row--) {
        tmp = dial_mask.ptr<uchar>(row);
        int left = 0;
        int right = dial_mask.cols - 1;
        while (left < right) {
            if (!tmp[left])      left++;
            if (!tmp[right])     right--;
            if (tmp[left] && tmp[right]) break;
        }

        if (diameter <= right - left) {
            diameter_index[0] = row;
            if (diameter < right - left) {
                diameter_index[1] = row;
            }
            diameter = right - left;
            diameter_index[2] = left;
            diameter_index[3] = right;  
        }
    }
    // std::cout << diameter_index[0] << " " <<diameter_index[1] << " " <<diameter_index[2] << " " <<diameter_index[3] << " " <<std::endl;

    if (diameter > 0) {
        center_location.x = (diameter_index[2] + diameter_index[3]) / 2;
        center_location.y = (diameter_index[0] + diameter_index[1]) / 2;
        return 0;
    }
    else {
        return -1;
    }
}
*/



// ��ȡͼ���������������С�����ת����ε��ĸ�����λ�á�
// ֮���������������Ҳ��Ϊ�˹��˵����ǵ����
int MeterReader::getMinAreaRectPoints(cv::Mat& pointer_mask, cv::Point2f* Ps) {

    //һ�����������ɸ������ĵ���б����ﶨ������������б�
    std::vector<std::vector<cv::Point>> contours;
    // ��������������ϵ
    std::vector<cv::Vec4i> hierarchy;
    // ����ģʽʹ�õ��ǽ������������
    cv::findContours(pointer_mask, contours, hierarchy,
        cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE, cv::Point());

    // �쳣�ж�
    int len = contours.size();
    if (len <= 0) return -1;

    int index = -1;
    int max_min_rect_area = 0; // ������С��Ӿ������
    for (int i = 0; i < len; i++) {
        cv::RotatedRect rect = cv::minAreaRect(contours[i]);
        if (rect.size.area() > max_min_rect_area) {
            max_min_rect_area = rect.size.area();
            index = i;
        }
    }

    cv::RotatedRect rect = cv::minAreaRect(contours[index]);
    rect.points(Ps);
    return 0;
}

// ��ȡָ��ͷ���ĵ㣨ָ����һ����ת���Σ�ͷ�����������㣬��������ȡ���Ǹõ�˳ʱ�뷽��Ϊ�̱ߵĵ㣩
// ���룺ָ��Ķ�ֵ��ͼ���Ǳ������ģ�ָ����ת�����ĸ�����
// �����ָ����ת�����ĸ�������ָ��ͷ�����������
int MeterReader::getPointerVertexIndex(cv::Point2f& center_location, cv::Point2f* Ps, int& vertex_index) {
    int max_distance = -1;
    for (int i = 0; i < 4; i++) {
        // ˳ʱ�뷽��Ϊ�̱�
        if (getDistance(Ps[i], Ps[(i + 1) % 4]) > getDistance(Ps[i], Ps[(i + 3) % 4])) {
            // ָ��ͷ��
            if (max_distance <= getDistance(Ps[i], center_location)) {
                max_distance = getDistance(Ps[i], center_location);
                vertex_index = i;
            }
        }
    }
    return 0;
}


int MeterReader::getPointerLocation(cv::Mat& pointer_mask, cv::Point2f& center_location, cv::Point2f* pointer_location) {

    // �ȸ���ָ���ֵ��ͼ�����ҵ���С�����ת���ζ���
    cv::Point2f Ps[4];
    int res = getMinAreaRectPoints(pointer_mask, Ps);
    if (res == -1) return res;

    // ���ݾ��ζ����ҵ�ͷ���ĵ������
    int vertex_index = -1;
    getPointerVertexIndex(center_location, Ps, vertex_index);

    // ��ָ���Ϊ�����֣������Ǳ����ĵĲ��ֶ�ֵ��Ϊ0
    cv::Point2f left_midPointer((Ps[vertex_index].x + Ps[(vertex_index + 1) % 4].x) / 2.0, (Ps[vertex_index].y + Ps[(vertex_index + 1) % 4].y) / 2.0);
    cv::Point2f right_midPointer((Ps[(vertex_index + 2) % 4].x + Ps[(vertex_index + 3) % 4].x) / 2.0, (Ps[(vertex_index + 2) % 4].y + Ps[(vertex_index + 3) % 4].y) / 2.0);

    std::vector<cv::Point2f> contour = {
            left_midPointer,
            Ps[(vertex_index + 1) % 4],
            Ps[(vertex_index + 2) % 4],
            right_midPointer
    };

    thresholdByContour(pointer_mask, pointer_mask, contour);

    // �ٸ��ݶ�ֵ�����ͼ����������С�����ת����
    res = getMinAreaRectPoints(pointer_mask, Ps);
    if (res == -1) return res;
    getPointerVertexIndex(center_location, Ps, vertex_index);

    cv::Point2f head_midPointer((Ps[vertex_index].x + Ps[(vertex_index + 3) % 4].x) / 2.0, (Ps[vertex_index].y + Ps[(vertex_index + 3) % 4].y) / 2.0);
    cv::Point2f tail_midPointer((Ps[(vertex_index + 1) % 4].x + Ps[(vertex_index + 2) % 4].x) / 2.0, (Ps[(vertex_index + 1) % 4].y + Ps[(vertex_index + 2) % 4].y) / 2.0);
    pointer_location[0] = head_midPointer;
    pointer_location[1] = tail_midPointer;
    return 0;
}


// �Ƕȷ����㵱ǰ�̶�ռ�̶ܿȵı���
float MeterReader::getAngleRatio(cv::Point2f* scale_locations, cv::Point2f& pointer_head_location, cv::Point2f& center_location) {

    // �̶ȿ�ʼ����x��������ļнǣ��̶Ƚ�������x��������ļнǣ��̶ȿ�ʼ����̶Ƚ�����ļн�
    float beginning_x_angle = atan2(center_location.y - scale_locations[0].y,
        scale_locations[0].x - center_location.x);
    float end_x_angle = atan2(center_location.y - scale_locations[1].y,
        scale_locations[1].x - center_location.x);
    float beginning_end_angle = 2 * CV_PI - (end_x_angle - beginning_x_angle);

    float pointer_x_angle = atan2(center_location.y - pointer_head_location.y,
        pointer_head_location.x - center_location.x);
    float beginning_pointer_angle = 0;
    if (pointer_head_location.y > center_location.y && pointer_head_location.x < center_location.x) {
        beginning_pointer_angle = pointer_x_angle - beginning_x_angle;
    }
    else {
        beginning_pointer_angle = 2 * CV_PI - (pointer_x_angle - beginning_x_angle);
    }

    float angleRatio = beginning_pointer_angle / beginning_end_angle;
    return angleRatio;
}

float MeterReader::getScaleValue(float angleRatio) {
    float range = SCALE_END - SCALE_BEGINNING;
    float value = range * angleRatio + SCALE_BEGINNING;
    return value;
}

void MeterReader::vis_for_test(cv::Mat& img, cv::Point2f* scale_locations, cv::Point2f* pointer_locations,
    cv::Point2f& center_location, float scale_value) {

    cv::Mat img_copy = img.clone();
    cv::Scalar red = cv::Scalar(0, 0, 255);
    cv::Scalar green = cv::Scalar(0, 255, 0);
    cv::Scalar blue = cv::Scalar(255, 0, 0);
    cv::circle(img_copy, scale_locations[0], 2, red, 2);
    cv::circle(img_copy, scale_locations[1], 2, red, 2);
    cv::circle(img_copy, center_location, 5, green, 2);

    cv::line(img_copy, pointer_locations[0], pointer_locations[1], blue, 3, 8);

    std::string scale_value_str = floatToString(scale_value);
    cv::putText(img_copy, scale_value_str, cv::Point(img.cols / 3, img.rows / 2),
        cv::FONT_HERSHEY_COMPLEX, 0.7, cv::Scalar(255, 0, 0), 2, 8);

    cv::imshow("vis for test", img_copy);
    cv::waitKey(0);
    // cv::waitKey(5000);
}

void MeterReader::drawMask(cv::Mat& img, cv::Mat& mask, float scale_value) {
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;

    cv::Mat dial_mask = cv::Mat(mask.rows, mask.cols, CV_8UC1);
    cv::Mat pointer_mask = cv::Mat(mask.rows, mask.cols, CV_8UC1);
    thresholdByCategory(mask, pointer_mask, 1);
    thresholdByCategory(mask, dial_mask, 2);

    cv::medianBlur(dial_mask, dial_mask, 7);
    cv::medianBlur(pointer_mask, pointer_mask, 7);

    cv::findContours(dial_mask, contours, hierarchy,
        cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE, cv::Point());
    cv::drawContours(img, contours, -1, cv::Scalar(0, 0, 255), 2, 8);

    cv::findContours(pointer_mask, contours, hierarchy,
        cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE, cv::Point());
    cv::drawContours(img, contours, -1, cv::Scalar(0, 255, 255), 2, 8);

    std::string scale_value_str = floatToString(scale_value);
    cv::putText(img, "value: " + scale_value_str, cv::Point(10, 50),
        cv::FONT_HERSHEY_COMPLEX, 0.6, cv::Scalar(255, 0, 0), 2, 8);
}


// ������mask, һ�����ĸ��������0-3�ֱ�����������̡�ָ��Ϳ̶�
// ����ǿ̶�ֵ
float MeterReader::reader_process(cv::Mat& img, cv::Mat& mask) {

    cv::Mat dial_mask = cv::Mat(mask.rows, mask.cols, CV_8UC1);
    cv::Mat pointer_mask = cv::Mat(mask.rows, mask.cols, CV_8UC1);
    thresholdByCategory(mask, pointer_mask, 1);
    thresholdByCategory(mask, dial_mask, 2);

    // �Ա���Ҫ������ʴ����
    cv::Mat kernel(5, 5, CV_8U, cv::Scalar(1));
    cv::erode(dial_mask, dial_mask, kernel);

    cv::Point2f scale_locations[2];
    getScaleLocation(dial_mask, scale_locations);

    cv::Point2f center_location;
    // getCenterLocation(dial_mask, center_location);
    getCenterLocation(img, center_location);

    cv::Point2f pointer_locations[2];
    getPointerLocation(pointer_mask, center_location, pointer_locations);

    float angle_ratio = getAngleRatio(scale_locations, pointer_locations[0], center_location);
    float scale_value = getScaleValue(angle_ratio);

    // ���ӻ����������ʹ��
    // vis_for_test(img, scale_locations, pointer_locations, center_location, scale_value);

    return scale_value;
}

// ��ȡ�������
std::vector<float> MeterReader::multi_read_process(const std::vector<cv::Mat>& input_image, const std::vector<cv::Mat>& output)
{
    int meter_num = input_image.size();
    std::cout << "meter_num: " << meter_num << std::endl;

    scale_values_.clear();

    for (int i_num = 0; i_num < meter_num; i_num++)
    {
        // std::cout << "ִ�е�������----------0������" << std::endl;

        image_ = input_image[i_num];
        // cv::imshow("image", image);
        // cv::waitKey(0);

        mask_ = output[i_num];
        // cv::imshow("mask", mask);
        // cv::waitKey(0);

        // std::cout << "ִ�е������ˡ�����" << std::endl;
        float scale_value = reader_process(image_, mask_);
        std::cout << "scale_value: " << scale_value << std::endl;
        scale_values_.push_back(scale_value);
    }

    // float scale_value = meterReader.reader_process(image, mask);
    // std::cout << "scale_value: " << scale_value << std::endl;
    // meterReader.drawMask(image, mask, scale_value);

    // const char* save_path = "C:/CPlusPlus/MetersReader/models/111_out.jpg";
    // cv::imwrite(save_path, image);
    // cv::imwrite(argv[4], person);
    return scale_values_;
}


cv::Mat MeterReader::result_visualizer(const cv::Mat& bgr, const std::vector<Object>& objects_remains, const std::vector<float> scale_values)
{
    cv::Mat output_image = bgr.clone();
    for (int i_results = 0; i_results < objects_remains.size(); i_results++)
    {
        cv::Rect bounding_box = objects_remains[i_results].rect;
        float result = scale_values[i_results];

        cv::Scalar color = cv::Scalar(237, 189, 101);
        cv::rectangle(output_image, bounding_box, color);  // Ŀ���

        std::string class_name = "Meter";
        cv::putText(output_image,
            class_name + " : " + std::to_string(result),
            cv::Point2d(bounding_box.x, bounding_box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
    }

    return output_image;
}