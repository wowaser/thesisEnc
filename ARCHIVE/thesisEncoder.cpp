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

#include "opencv2/videoio.hpp"

//#include <windows.h>
//#pragma comment(lib, "ws2_32.lib")

simplelogger::Logger* logger = simplelogger::LoggerFactory::CreateConsoleLogger();

const char* video = "../vids/4kvideo.mp4";

int w = 3840;
int h = 2160;


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




void encode(CUdevice cuDevice, bool lowLatency) {

	uvgrtp::context ctx;
	uvgrtp::session* sess = ctx.create_session("127.0.0.1");
	uvgrtp::media_stream* sender = sess->create_stream(8888, 8889, RTP_FORMAT_H264, RCE_NO_FLAGS);

	NV_ENC_BUFFER_FORMAT bufFormat_ = NV_ENC_BUFFER_FORMAT_IYUV; // which one?

	CUcontext cuContext = NULL;


	NV_ENC_INITIALIZE_PARAMS initializeParams = { NV_ENC_INITIALIZE_PARAMS_VER };
	NV_ENC_CONFIG encodeConfig = { NV_ENC_CONFIG_VER };
	initializeParams.encodeConfig = &encodeConfig;

	NV_ENC_PIC_PARAMS picParams = { NV_ENC_PIC_PARAMS_VER };

	ck(cuCtxCreate(&cuContext, 0, cuDevice));

	NvEncoderCuda enc(cuContext, w, h, NV_ENC_BUFFER_FORMAT_IYUV, 1);

	enc.CreateDefaultEncoderParams(
		&initializeParams,
		NV_ENC_CODEC_H264_GUID,
		NV_ENC_PRESET_P1_GUID, // what is this?
		NV_ENC_TUNING_INFO_LOW_LATENCY); // what does this mean?

	initializeParams.enablePTD = 1;

	initializeParams.enableSubFrameWrite = 1; // Speedup, explain why, maybe exlude that shit, maybe ref the FPGA macroblock pipeline or whatever

	encodeConfig.rcParams.averageBitRate = 30 * 1000 * 1000; // Why this?
	encodeConfig.rcParams.rateControlMode = NV_ENC_PARAMS_RC_CBR; // GOOD
	encodeConfig.rcParams.multiPass = NV_ENC_MULTI_PASS_DISABLED; // GOOD

	initializeParams.frameRateNum = 120;
	initializeParams.frameRateDen = 1;

	encodeConfig.rcParams.vbvBufferSize = static_cast<uint32_t>( // GOOD
		static_cast<float>(encodeConfig.rcParams.averageBitRate) 
		/ 
		static_cast<float>(initializeParams.frameRateNum)
	);
	encodeConfig.rcParams.vbvInitialDelay = encodeConfig.rcParams.vbvBufferSize;

	encodeConfig.gopLength = NVENC_INFINITE_GOPLENGTH; // why?  

	encodeConfig.frameIntervalP = 1; // GOOD

	encodeConfig.encodeCodecConfig.h264Config.repeatSPSPPS = 1; // whatever

	encodeConfig.encodeCodecConfig.h264Config.enableIntraRefresh = 1; // GOOD
	encodeConfig.encodeCodecConfig.h264Config.intraRefreshPeriod = 100;
	//encodeConfig.encodeCodecConfig.h264Config.singleSliceIntraRefresh = 1;
	encodeConfig.encodeCodecConfig.h264Config.intraRefreshCnt = 10;

	encodeConfig.rcParams.enableNonRefP = 1; // what is that?



	enc.CreateEncoder(&initializeParams);

	int counter = 0;
	std::chrono::milliseconds dur_ = std::chrono::milliseconds(0);

	try
	{
		int frameSize = enc.GetFrameSize();
		std::unique_ptr<uint8_t[]> pHostFrame(new uint8_t[frameSize]);

		std::streamsize nRead = 0;

		std::ifstream inputFile("C:/Users/vladimir/Desktop/vid.yuv", std::ifstream::in | std::ifstream::binary);
		if (!inputFile)
		{
			throw std::exception("can't open file");
		}

		do {
			std::vector<std::vector<uint8_t>> vPacket;
			nRead = inputFile.read(reinterpret_cast<char*>(pHostFrame.get()), frameSize).gcount();
			auto start = std::chrono::high_resolution_clock::now();
			if (nRead == frameSize)
			{	
				const NvEncInputFrame* encoderInputFrame = enc.GetNextInputFrame();

				NvEncoderCuda::CopyToDeviceFrame(cuContext, pHostFrame.get(), 0, (CUdeviceptr)encoderInputFrame->inputPtr,
					(int)encoderInputFrame->pitch,
					enc.GetEncodeWidth(),
					enc.GetEncodeHeight(),
					CU_MEMORYTYPE_HOST,
					encoderInputFrame->bufferFormat,
					encoderInputFrame->chromaOffsets,
					encoderInputFrame->numChromaPlanes);

				enc.EncodeFrame(vPacket);
			}
			else
			{
				enc.EndEncode(vPacket);
			}
			// Finishing time.
			auto stop = std::chrono::high_resolution_clock::now();
			auto time = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

			// Printing the measured time.
			//std::cout << "encoding: " << time.count() << " ms" << std::endl;
			dur_ += time;
			++counter;


			for (std::vector<uint8_t>& packet : vPacket) {

				// Allow time point to overflow, since it's symmetric on the other side
				//auto st = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch());
				//uint32_t ts = static_cast<uint32_t>(st.count());



				sender->push_frame(packet.data(), packet.size(), createTs(), RTP_NO_H26X_SCL);
			}
		} while (nRead == frameSize);

		enc.DestroyEncoder();
		inputFile.close();
		ctx.destroy_session(sess);
	}
	catch (const std::exception& e)
	{
		std::cout << e.what() << std::endl;
	}

	auto resTime = std::chrono::duration_cast<std::chrono::milliseconds>(dur_);
	std::cout << "avg encoding: " << resTime.count() / float(counter) << " ms" << std::endl;
	
}

void decode(CUdevice cudaDevice_) {

	uvgrtp::context ctx;
	uvgrtp::session* sess = ctx.create_session("127.0.0.1");

	uvgrtp::media_stream* receiver = sess->create_stream(8889, 8888, RTP_FORMAT_H264, RCE_NO_FLAGS);

	int nFrame = 0;
	CUdeviceptr dpRgbFrame = 0;
	std::chrono::milliseconds dur_ = std::chrono::milliseconds(0);
	try
	{
		CUcontext cuContext = NULL;
		ck(cuCtxCreate(&cuContext, 0, cudaDevice_));


		NvDecoder dec(cuContext, false, cudaVideoCodec_H264, true);
		//NvDecoder dec(cuContext, false, cudaVideoCodec_H264, true, false, nullptr, nullptr, 0, 0, 1000U, true);

		uint8_t* pVideo = NULL;

		std::ofstream outputFile("C:/Users/vladimir/Desktop/outMine.yuv", std::ios::out | std::ios::binary);
		if (!outputFile)
		{
			throw std::exception("can't open output file");
		}

		do {
			auto frame = receiver->pull_frame();

			//auto end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch());
			//uint32_t ts = static_cast<uint32_t>(end.count());

			//std::cout << "system latency: " << ts - frame->header.timestamp << std::endl;

			std::cout << "send latency: " << createTs() - frame->header.timestamp << "\n\n";
			
			if (!frame) {
				std::cout << "pull frame got nullptr!" << std::endl;
				continue;
			}

			auto start = std::chrono::high_resolution_clock::now();
			int framesReturned = dec.Decode(frame->payload, static_cast<int>(frame->payload_len), CUVID_PKT_ENDOFPICTURE);
			auto stop = std::chrono::high_resolution_clock::now();
			auto time = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

			// Printing the measured time.
			//std::cout << "decoding: " << time.count() << " ms" << std::endl;
			dur_ += time;

			uint8_t* pFrame;
			for (int i = 0; i < framesReturned; i++) {
				std::cout << framesReturned << std::endl;
				pFrame = dec.GetFrame();
				outputFile.write(reinterpret_cast<char*>(pFrame), dec.GetFrameSize());
				nFrame++;
			}

			uvgrtp::frame::dealloc_frame(frame);
		} while (nFrame != 600);

		outputFile.close();

		/*std::cout << "Total frame decoded: " << nFrame << std::endl
			<< "Saved in file " << szOutFilePath << " in "
			<< (eOutputFormat == native ? (dec.GetBitDepth() == 8 ? "nv12" : "p010") : (eOutputFormat == bgra ? "bgra" : "bgra64"))
			<< " format" << std::endl;*/
	}
	catch (const std::exception& e)
	{
		std::cout << e.what() << std::endl;
	}
	auto resTime = std::chrono::duration_cast<std::chrono::milliseconds>(dur_);
	std::cout << "avg decoding: " << resTime.count() / float(nFrame) << " ms" << std::endl;

}

int main()
{
	ck(cuInit(0));

	CUdevice cudaDevice = 0;
	ck(cuDeviceGet(&cudaDevice, 0));


	NvThread decProc(std::thread(encode, cudaDevice, true));
	NvThread encProc(std::thread(decode, cudaDevice));

	encProc.join();
	decProc.join();
}

