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

#include "opencv2/video.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudawarping.hpp"
#include <opencv2/core/opengl.hpp>
#include <opencv2/cudacodec.hpp>

//#include <windows.h>
//#pragma comment(lib, "ws2_32.lib")

simplelogger::Logger* logger = simplelogger::LoggerFactory::CreateConsoleLogger();

const char* video = "../vids/4kvideo.mp4";


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




void createEncParams(bool lowLatency, NV_ENC_INITIALIZE_PARAMS* initializeParams)
{
	NV_ENC_CONFIG* encodeConfig = initializeParams->encodeConfig;

	initializeParams->enablePTD = 1;

	initializeParams->enableSubFrameWrite = 1; // Speedup, explain why, maybe exlude that shit, maybe ref the FPGA macroblock pipeline or whatever

	encodeConfig->rcParams.averageBitRate = 30 * 1000 * 1000; // Why this?
	encodeConfig->rcParams.rateControlMode = NV_ENC_PARAMS_RC_CBR; // GOOD
	encodeConfig->rcParams.multiPass = NV_ENC_MULTI_PASS_DISABLED; // GOOD

	initializeParams->frameRateNum = 30;
	initializeParams->frameRateDen = 1;

	encodeConfig->rcParams.vbvBufferSize = static_cast<uint32_t>( // GOOD
		static_cast<float>(encodeConfig->rcParams.averageBitRate)
		/
		static_cast<float>(initializeParams->frameRateNum)
		);
	encodeConfig->rcParams.vbvInitialDelay = encodeConfig->rcParams.vbvBufferSize;

	encodeConfig->gopLength = NVENC_INFINITE_GOPLENGTH; // why?  

	encodeConfig->frameIntervalP = 1; // GOOD

	encodeConfig->encodeCodecConfig.h264Config.repeatSPSPPS = 1; // whatever

	encodeConfig->encodeCodecConfig.h264Config.enableIntraRefresh = 1; // GOOD
	encodeConfig->encodeCodecConfig.h264Config.intraRefreshPeriod = 100;
	encodeConfig->encodeCodecConfig.h264Config.singleSliceIntraRefresh = 1;
	encodeConfig->encodeCodecConfig.h264Config.intraRefreshCnt = 10;

	encodeConfig->rcParams.enableNonRefP = 1; // what is that?

}

int main()
{

	int width = 1920;
	int height = 1080;

	// Choose setting for comparison
	bool lowLatency = true;

	// Cuda init
	ck(cuInit(0));
	CUdevice cudaDevice = 0;
	ck(cuDeviceGet(&cudaDevice, 0));
	CUcontext cuContext = NULL;
	ck(cuCtxCreate(&cuContext, 0, cudaDevice));

	// Init NVENC vars
	NV_ENC_INITIALIZE_PARAMS initializeParams = { NV_ENC_INITIALIZE_PARAMS_VER };
	NV_ENC_CONFIG encodeConfig = { NV_ENC_CONFIG_VER };
	NV_ENC_PIC_PARAMS picParams = { NV_ENC_PIC_PARAMS_VER };
	initializeParams.encodeConfig = &encodeConfig;

	std::unique_ptr<NvEncoderCuda> enc = std::make_unique<NvEncoderCuda>(cuContext, width, height, NV_ENC_BUFFER_FORMAT_ABGR, 1);

	enc->CreateDefaultEncoderParams(
		&initializeParams,
		NV_ENC_CODEC_H264_GUID,
		NV_ENC_PRESET_P1_GUID, // what is this?
		NV_ENC_TUNING_INFO_LOW_LATENCY); // what does this mean?

	createEncParams(lowLatency, &initializeParams);
	enc->CreateEncoder(&initializeParams);

	// Streaming library
	uvgrtp::context ctx;
	uvgrtp::session* sess = ctx.create_session("192.168.225.39");
	uvgrtp::media_stream* sender = sess->create_stream(8888, 8889, RTP_FORMAT_H264, RCE_NO_FLAGS);



	cv::VideoCapture cap(0);
	cap.open(0);

	if (!cap.isOpened()) { return -1; }

	cap.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
	cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);

	cv::Mat frame;
	cv::cuda::GpuMat gFrame;

	//cv::namedWindow("GPU", cv::WINDOW_OPENGL);



	int counter = 0;
	std::chrono::milliseconds dur_ = std::chrono::milliseconds(0);

	while (true) {

		// Acquisition
		cap >> frame;
		gFrame.upload(frame);
		cv::cuda::cvtColor(gFrame, gFrame, cv::ColorConversionCodes::COLOR_BGR2RGBA);


		auto start = std::chrono::high_resolution_clock::now();
		// Encoding
		const NvEncInputFrame* encoderInputFrame = enc->GetNextInputFrame();
		std::vector<std::vector<uint8_t>> vPacket;
		NvEncoderCuda::CopyToDeviceFrame(cuContext, gFrame.cudaPtr(), (uint32_t)gFrame.step, (CUdeviceptr)encoderInputFrame->inputPtr,
			(int)encoderInputFrame->pitch,
			enc->GetEncodeWidth(),
			enc->GetEncodeHeight(),
			CU_MEMORYTYPE_DEVICE,
			encoderInputFrame->bufferFormat,
			encoderInputFrame->chromaOffsets,
			encoderInputFrame->numChromaPlanes);

		if (counter % 300 == 0) {
			picParams.encodePicFlags = NV_ENC_PIC_FLAG_FORCEIDR | NV_ENC_PIC_FLAG_OUTPUT_SPSPPS;
		}

		enc->EncodeFrame(vPacket, &picParams);

		// Finishing time.
		auto stop = std::chrono::high_resolution_clock::now();
		auto time = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

		// Printing the measured time.
		//std::cout << "encoding: " << time.count() << " ms" << std::endl;
		dur_ += time;
		++counter;



		for (std::vector<uint8_t>& packet : vPacket) {
			sender->push_frame(packet.data(), packet.size(), createTs(), RTP_NO_H26X_SCL);
		}
	}

	ctx.destroy_session(sess);
	enc->DestroyEncoder();
}