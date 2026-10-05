#include <GxIAPI.h>
#include <DxImageProc.h>
#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/async.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include "DxImageProc.h"
#include <spdlog/sinks/stdout_color_sinks.h>
#ifndef BYTE
typedef unsigned char BYTE;
#endif

// 全局句柄与同步标志
GX_DEV_HANDLE g_device = nullptr;
std::atomic<bool> g_grabbing{false};
std::mutex g_frame_mutex;
cv::Mat g_latest_frame;

void init_logging() {
    spdlog::init_thread_pool(8192, 1);
    auto file_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
        "logs/dh_camera.log", 5 * 1024 * 1024, 3);
    auto console = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    auto logger = std::make_shared<spdlog::async_logger>(
        "dh", spdlog::sinks_init_list{console, file_sink},
        spdlog::thread_pool(), spdlog::async_overflow_policy::overrun_oldest);
    logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%t] [%^%l%$] %v");
    logger->set_level(spdlog::level::debug);
    spdlog::set_default_logger(logger);
}

// 1. 确保回调函数的签名是正确的 C API 格式
void GX_STDC OnFrameCallback(GX_FRAME_CALLBACK_PARAM* pFrame) {
    
    // 2. 错误状态判断
    if (pFrame->status != GX_FRAME_STATUS_SUCCESS) {
        SPDLOG_WARN("帧采集异常, 状态码: {}", pFrame->status);
        return;
    }

    // 3. 图像转换逻辑必须写在这里（{} 内部）
    cv::Mat frame;
    
    // ⚠️ 注意：C API 中字段名可能是  nWidth，请务必根据头文件提示修改！
    // 如果按下 Ctrl+单击 GX_FRAME_CALLBACK_PARAM 进去看，它通常会包含一个 stFrameInfo 成员。
    
    // 假设你现在的代码是直接 pFrame->nWidth，如果报错，请改为 pFrame-> nWidth
    if (pFrame-> nPixelFormat == GX_PIXEL_FORMAT_BAYER_RG8) {
        cv::Mat raw(pFrame->nHeight, 
                    pFrame->nWidth, 
                    CV_8UC1, 
                    const_cast<void*>(pFrame->pImgBuf));
        cv::cvtColor(raw, frame, cv::COLOR_BayerRG2BGR);
    } 
    else if (pFrame-> nPixelFormat == GX_PIXEL_FORMAT_MONO8) {
        cv::Mat raw(pFrame->nHeight, 
                    pFrame->nWidth, 
                    CV_8UC1, 
                    const_cast<void*>(pFrame->pImgBuf));
        frame = raw.clone();
    }

    // 4. 图像入队
    if (!frame.empty()) {
        std::lock_guard<std::mutex> lock(g_frame_mutex);
        g_latest_frame = frame.clone(); 
    }

    BYTE* pRGB24Buf = new BYTE[pFrame->nWidth * pFrame->nHeight * 3];
    VxInt32 dxStatus = DxRaw8toRGB24(
        (BYTE*)pFrame->pImgBuf,      // 原始 Bayer 数据
        pRGB24Buf,                    // 输出 RGB24 缓冲区
        pFrame->nWidth,
        pFrame->nHeight,
        RAW2RGB_NEIGHBOUR,            // 插值算法：邻域插值
        DX_PIXEL_COLOR_FILTER(BAYERRG), // Bayer 排列：RGGB
        false                         // 不翻转
    );

    if (dxStatus == DX_OK) {
        cv::Mat frame(pFrame->nHeight, pFrame->nWidth, CV_8UC3, pRGB24Buf);
        std::lock_guard<std::mutex> lock(g_frame_mutex);
        g_latest_frame = frame.clone();
    }
    delete[] pRGB24Buf;
    
} 

bool init_camera() {
    // 1. 初始化库
    GX_STATUS status = GXInitLib();
    if (status != GX_STATUS_SUCCESS) {
        SPDLOG_ERROR("GXInitLib 失败, 状态码: {}", status);
        return false;
    }

    // 2. 枚举设备
    uint32_t device_num = 0;
    status = GXUpdateDeviceList(&device_num, 1000);
    if (status != GX_STATUS_SUCCESS || device_num == 0) {
        SPDLOG_WARN("未找到大恒相机设备");
        return false;
    }
    SPDLOG_INFO("找到 {} 台设备", device_num);

    // 3. 打开第一台设备
    status = GXOpenDeviceByIndex(1, &g_device); // 索引从 1 开始
    if (status != GX_STATUS_SUCCESS) {
        SPDLOG_ERROR("打开设备失败, 状态码: {}", status);
        return false;
    }

    // 4. 注册回调（C API 核心）
    status = GXRegisterCaptureCallback(g_device, nullptr, OnFrameCallback);
    if (status != GX_STATUS_SUCCESS) {
        SPDLOG_ERROR("注册回调失败, 状态码: {}", status);
        return false;
    }

    // 5. 发送开采命令
    status = GXSendCommand(g_device, GX_COMMAND_ACQUISITION_START);
    if (status != GX_STATUS_SUCCESS) {
        SPDLOG_ERROR("开始采集失败, 状态码: {}", status);
        return false;
    }

    g_grabbing = true;
    SPDLOG_INFO("开始采集");
    return true;
}

void cleanup_camera() {
    if (g_grabbing) {
        GXSendCommand(g_device, GX_COMMAND_ACQUISITION_STOP);
        GXUnregisterCaptureCallback(g_device);
        SPDLOG_INFO("已停止采集");
    }
    if (g_device) {
        GXCloseDevice(g_device);
        g_device = nullptr;
        SPDLOG_INFO("相机已关闭");
    }
    GXCloseLib();
    SPDLOG_INFO("SDK 已释放");
}

int main() {
    init_logging();
    SPDLOG_INFO("=== 大恒相机采集程序启动 ===");

    if (!init_camera()) {
        SPDLOG_CRITICAL("相机初始化失败");
        return -1;
    }

    SPDLOG_INFO("按 ESC 退出...");
    while (g_grabbing) {
        {
            std::lock_guard<std::mutex> lock(g_frame_mutex);
            if (!g_latest_frame.empty()) {
                cv::imshow("Daheng Camera", g_latest_frame);
            }
        }
        if (cv::waitKey(1) == 27) break;
    }

    cleanup_camera();
    SPDLOG_INFO("=== 程序正常退出 ===");
    spdlog::shutdown();
    return 0;
}