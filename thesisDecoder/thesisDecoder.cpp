// uvgrtp
#include "uvgrtp/lib.hh"

#include <iostream>
#include <chrono>

#include <cuda.h>

#include "NvDecoder/NvDecoder.h"
#include "NvEncoder/NvEncoderCuda.h"
#include "../Utils/NvEncoderCLIOptions.h"
#include "../Utils/NvCodecUtils.h"
//#include "../Utils/FFmpegStreamer.h"
//#include "../Utils/FFmpegDemuxer.h"
#include "../Utils/ColorSpace.h"

// OpenCV

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>

//#include <windows.h>
//#pragma comment(lib, "ws2_32.lib")

simplelogger::Logger* logger = simplelogger::LoggerFactory::CreateConsoleLogger();

std::chrono::system_clock::time_point createEpoch()
{
	/* Create stable epoch from 1.1.2000 */
	std::tm t = std::tm();
	t.tm_year = 100;
	t.tm_mon = 0;
	t.tm_mday = 1;
	std::time_t tt = std::mktime(&t);
	return std::chrono::system_clock::from_time_t(tt);
}

uint32_t createTs()
{
	std::chrono::system_clock::duration d = std::chrono::system_clock::now() - createEpoch();
	auto us = std::chrono::duration_cast<std::chrono::milliseconds>(d);
	return static_cast<uint32_t>(us.count());
}


int main()
{
	ck(cuInit(0));
	CUdevice cudaDevice = 0;
	ck(cuDeviceGet(&cudaDevice, 0));
	CUcontext cuContext = NULL;
	ck(cuCtxCreate(&cuContext, 0, cudaDevice));

	std::unique_ptr<NvDecoder> dec = std::make_unique<NvDecoder>(cuContext, true, cudaVideoCodec_H264, true);
	dec->SetOperatingPoint(0, 0);

	uvgrtp::context ctx;
	uvgrtp::session* sess = ctx.create_session("192.168.225.6");
	uvgrtp::media_stream* receiver = sess->create_stream(8889, 8888, RTP_FORMAT_H264, RCE_NO_FLAGS);


	int nFrame = 0;
	std::chrono::milliseconds dur_ = std::chrono::milliseconds(0);
	CUdeviceptr clrGpuPtr = 0;

	bool first = true;
	while (true) {
		auto frame = receiver->pull_frame();

		if (!frame) {
			std::cout << "pulled zero" << std::endl;
			continue;
		}

		//std::cout <<"frame: " <<  frame->header.timestamp << std::endl;
		//std::cout << "us: " << createTs() << std::endl;

		std::cout << "latency: " << createTs() - frame->header.timestamp << "\n\n";

		auto start = std::chrono::high_resolution_clock::now();
		int framesReturned = dec->Decode(frame->payload, static_cast<int>(frame->payload_len), CUVID_PKT_ENDOFPICTURE);
		auto stop = std::chrono::high_resolution_clock::now();
		auto time = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
		// Printing the measured time.
		//std::cout << "decoding: " << time.count() << " ms" << std::endl;
		dur_ += time;
		nFrame++;

		if (framesReturned > 0) {
			int w = dec->GetWidth();
			int h = dec->GetHeight();
			if (first) {
				ck(cuCtxSetCurrent(cuContext));
				ck(cuMemAlloc(&clrGpuPtr, w * h * 4));
				first = false;
			}
			auto frameInfo = dec->GetVideoFormatInfo();

			Nv12ToColor32<BGRA32>(dec->GetFrame(), w, (uint8_t*)clrGpuPtr, 4 * w, w, h, frameInfo.video_signal_description.matrix_coefficients);
			cv::cuda::GpuMat gpuFrame = cv::cuda::GpuMat(h, w, CV_8UC4, (uint8_t*)clrGpuPtr);

			if (!gpuFrame.empty()) {
				cv::namedWindow("output", cv::WINDOW_OPENGL);
				cv::imshow("output", gpuFrame);
				if (cv::waitKey(1) == 27) break;
			}
		}
		else {
			std::cout << "frames returned 0" << std::endl;
		}
	}

	ctx.destroy_session(sess);

	auto resTime = std::chrono::duration_cast<std::chrono::milliseconds>(dur_);
	std::cout << "avg decoding: " << resTime.count() / float(nFrame) << " ms" << std::endl;
}
