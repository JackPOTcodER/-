#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

using namespace cv;
using namespace std;

const float ARMOR_WIDTH = 135.0f;
const float ARMOR_HEIGHT = 125.0f;
const float LIGHT_BAR_LENGTH = 52.0f;

Mat cameraMatrix = (Mat_<double>(3, 3) <<
    1000.0, 0.0, 640.0, 
    0.0, 1000.0, 360.0,
    0.0, 0.0, 1.0
);

Mat distCoeffs = (Mat_<double>(1, 5) << 0.0, 0.0, 0.0, 0.0, 0.0);
 

// 灯条结构体，用于存储筛选出的灯条信息
struct LightBar {
    RotatedRect rect; // 旋转矩形
    Point2f center;   // 中心点
    float length;     // 长度
    float width;      // 宽度
    float angle;      // 倾斜角度
    vector<Point2f> corners;
};

float getAngleDiff(float a1, float a2)
{
    float diff = abs(a1 - a2);
    if(diff > 90.0f) diff = 180.0f - diff;
    return diff;
}

// 全局参数（可根据实际相机调整）
const int BINARY_THRESHOLD = 150; // 二值化阈值
const float MIN_ASPECT_RATIO = 1.5; // 灯条最小长宽比
const float MAX_ASPECT_RATIO = 6.0; // 灯条最大长宽比
const float MIN_AREA = 20.0;        // 灯条最小面积
const float MAX_AREA = 5000.0;      // 灯条最大面积

// 预处理：分离颜色并二值化
Mat preprocessImage(const Mat& src, bool is_red) 
{
    Mat gray, binary;
    
    // 1. 颜色通道分离（BGR信息利用）
    vector<Mat> channels;
    split(src, channels); // 分离为 B, G, R 三个通道
    
    if (is_red) {
        // 识别红色灯条：红色在 R 通道高，在 B 通道低
        // 常用方法：R - B 或者直接使用 R 通道减去 B 通道
        subtract(channels[2], channels[0], gray); 
    } else {
        // 识别蓝色灯条：蓝色在 B 通道高，在 R 通道低
        // 常用方法：B - R
        subtract(channels[0], channels[2], gray);
    }

    // 2. 二值化（将灰度图转为黑白二值图）
    threshold(gray, binary, BINARY_THRESHOLD, 255, THRESH_BINARY);

        // 3. 形态学操作：开运算去噪，闭运算连接断裂区域
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(binary, binary, MORPH_OPEN, kernel);
    morphologyEx(binary, binary, MORPH_CLOSE, kernel);

    return binary;
}

// 筛选灯条
vector<LightBar> findLightBars(const Mat& binary) 
{
    vector<vector<Point>> contours;
    // 寻找外轮廓
    findContours(binary, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    vector<LightBar> light_bars;
    for (const auto& contour : contours) {
        float area = contourArea(contour);
        // 1. 面积过滤
        if (area < 50.0f || area > 5000.0f) continue;

        // 2. 拟合旋转矩形
        RotatedRect rect = minAreaRect(contour);
        float w = rect.size.width;
        float h = rect.size.height;
        float length = max(w, h);
        float width = min(w, h);

        // 3. 长宽比过滤 (灯条通常是细长的)
        float aspect_ratio = length / width;
        if (aspect_ratio < 2.0f || aspect_ratio > 10.0f) continue;

    // 4. 填充灯条信息
        LightBar bar;
        bar.rect = rect;
        bar.center = rect.center;
        bar.length = length;
        bar.width = width;
        bar.angle = rect.angle;
        
        // 绘制识别到的灯条（调试用）
        Point2f vertices[4];
        rect.points(vertices);
        for (int i = 0; i < 4; i++) 
            bar.corners.push_back(vertices[i]);

        light_bars.push_back(bar);
    }
    return light_bars;
}

vector<pair<Point2f, Point2f>> matchArmors(const vector<LightBar>& light_bars) {
    vector<pair<Point2f, Point2f>> matched_armors;

    for (size_t i = 0; i < light_bars.size(); i++) {
        for (size_t j = i + 1; j < light_bars.size(); j++) {
            const LightBar& bar1 = light_bars[i];
            const LightBar& bar2 = light_bars[j];

            // 1. 平行度检查 (图纸要求灯条应是平行的)
            float angle_diff = getAngleDiff(bar1.angle, bar2.angle);
            if (angle_diff > 10.0f) continue; // 允许10度误差

            // 2. 中心距检查 (装甲板灯条间距大概在几十到一百多像素之间)
            float center_distance = norm(bar1.center - bar2.center);
            float avg_length = (bar1.length + bar2.length) / 2.0f;
            if (center_distance < avg_length * 1.5f || center_distance > avg_length * 6.0f) continue;

            // 3. 高度差检查 (两个灯条中心Y坐标不能差太多)
            if (abs(bar1.center.y - bar2.center.y) > avg_length * 0.8f) continue;

            // 4. 极其关键的步骤：PnP 解算与物理尺寸验证
            // 构建 3D 物理坐标 (单位: mm)
            // 这里我们假设装甲板是一个矩形平面，四个角点对应灯条的中心线
            // 考虑到图纸的倾斜角度，这里我们使用一个理想化的矩形进行验证
            vector<Point3f> objectPoints = {
                Point3f(-ARMOR_WIDTH / 2.0f, -ARMOR_HEIGHT / 2.0f, 0),
                Point3f( ARMOR_WIDTH / 2.0f, -ARMOR_HEIGHT / 2.0f, 0),
                Point3f( ARMOR_WIDTH / 2.0f,  ARMOR_HEIGHT / 2.0f, 0),
                Point3f(-ARMOR_WIDTH / 2.0f,  ARMOR_HEIGHT / 2.0f, 0)
            };

            // 获取图像上的 2D 角点 (简化处理：使用两个灯条的端点构建矩形)
            // 真实项目中应根据灯条方向确定四个角点的正确顺序
            vector<Point2f> imagePoints;
            // 这里做简化处理，实际需要根据灯条方向精确选取
            imagePoints.push_back(bar1.center + Point2f(-bar1.width/2, -bar1.length/2));
            imagePoints.push_back(bar2.center + Point2f( bar2.width/2, -bar2.length/2));
            imagePoints.push_back(bar2.center + Point2f( bar2.width/2,  bar2.length/2));
            imagePoints.push_back(bar1.center + Point2f(-bar1.width/2,  bar1.length/2));

            Mat rvec, tvec;
            // 调用 solvePnP 解算位姿
            bool success = solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec);
            if (!success) continue;

            // 验证解算结果
            double distance = norm(tvec); // 距离 (mm)
            if (distance < 200.0 || distance > 10000.0) continue; // 距离过滤，太近太远都不合理

            // 可以进一步验证解算出的宽度是否接近 135mm
            // 这里推荐一个更简单的几何验证：根据距离和焦距反推像素宽度
            // 如果图像上测量的像素宽度换算成实际物理宽度严重偏离 135mm，则剔除
            float pixel_width = center_distance; // 两个灯条中心点的像素距离
            float estimated_physical_width = (pixel_width * distance) / cameraMatrix.at<double>(0, 0);
            
            // 允许 30% 的误差范围
            if (abs(estimated_physical_width - ARMOR_WIDTH) > ARMOR_WIDTH * 0.3f) continue;

            // 通过所有验证，配对成功
            matched_armors.push_back({bar1.center, bar2.center});
        }
    }
    return matched_armors;
}


int main() {
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Error: Could not open camera." << endl;
        return -1;
    }

    Mat frame, binary;
    bool is_red = true; // 根据目标颜色切换

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        // 1. 预处理
        binary = preprocessImage(frame, is_red);

        // 2. 寻找灯条
        vector<LightBar> light_bars = findLightBars(binary);

        // 3. 配对并进行物理尺寸与姿态验证
        vector<pair<Point2f, Point2f>> matched_armors = matchArmors(light_bars);

        // 4. 可视化
        for (const auto& match : matched_armors) {
            line(frame, match.first, match.second, Scalar(0, 255, 0), 2); // 画连线
            circle(frame, match.first, 5, Scalar(0, 0, 255), -1); // 画灯条中心
            circle(frame, match.second, 5, Scalar(0, 0, 255), -1);
        }

        imshow("Armor Detection", frame);
        imshow("Binary", binary);

        if (waitKey(30) == 27) break;
    }

    cap.release();
    destroyAllWindows();
    return 0;
}