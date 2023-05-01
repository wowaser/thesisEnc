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
#include <queue>
#include <mutex>
#include <atomic>
#include <thread>

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



static void __receiveHook(void* arg, uvg_rtp::frame::rtp_frame* frame);

class decoder {
public:
	decoder(CUdevice cudaDevice, std::atomic_bool& run):cudaDevice_(cudaDevice), run_(run) {

		sess = ctx.create_session("127.0.0.1");
		stream = sess->create_stream(8889, 8888, RTP_FORMAT_H264, RCE_NO_FLAGS);

		if (stream->install_receive_hook(this, __receiveHook) != RTP_OK) {
			throw std::exception("failed to install rtp receiver hook");
		}

		ck(cuCtxCreate(&cuContext_, 0, cudaDevice_));
		dec = std::make_unique<NvDecoder>(cuContext_, false, cudaVideoCodec_H264, true);
		
		decodeThread_ = std::thread(&decoder::decodeLoop, this);

	};

	~decoder() {
		if (decodeThread_.joinable())
			decodeThread_.join();

		ctx.destroy_session(sess);
	}

	void setLatestFrame(uvg_rtp::frame::rtp_frame* frame) {
		{
			std::lock_guard<std::mutex> guard(decodeMtx_);
			q_.push(frame);
		}
		decodeCondition_.notify_one();
	};

private:
	CUdevice cudaDevice_ = NULL;
	std::queue<uvgrtp::frame::rtp_frame*> q_;
	mutable std::mutex decodeMtx_;
	std::condition_variable decodeCondition_;

	std::thread decodeThread_;
	std::unique_ptr<NvDecoder> dec;
	CUcontext cuContext_ = NULL;
	std::atomic_bool& run_;

	uvgrtp::context ctx;
	uvgrtp::session* sess = nullptr;

	uvgrtp::media_stream* stream = nullptr;


	void decodeLoop(){

		std::ofstream outputFile("C:/Users/vladimir/Desktop/outMine.yuv", std::ios::out | std::ios::binary);
		if (!outputFile)
		{
			throw std::exception("can't open output file");
		}
		std::chrono::milliseconds dur_ = std::chrono::milliseconds(0);
		int nFrame = 0;

		while (run_)
		{
			std::unique_lock<std::mutex> lock(decodeMtx_);
			decodeCondition_.wait(lock, [this] {return !this->q_.empty(); });

			auto ptr = q_.front();
			q_.pop();

			lock.unlock();

			std::cout << createTs() - ptr->header.timestamp << std::endl;

			auto start = std::chrono::high_resolution_clock::now();
			int framesReturned = dec->Decode(ptr->payload, static_cast<int>(ptr->payload_len), CUVID_PKT_ENDOFPICTURE);
			auto stop = std::chrono::high_resolution_clock::now();
			auto time = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

			// Printing the measured time.
			//std::cout << "decoding: " << time.count() << " ms" << std::endl;
			dur_ += time;

			uint8_t* pFrame;
			for (int i = 0; i < framesReturned; i++) {
				pFrame = dec->GetFrame();
				outputFile.write(reinterpret_cast<char*>(pFrame), dec->GetFrameSize());
				nFrame++;
			}

			uvgrtp::frame::dealloc_frame(ptr);
		}
		//std::cout << "avg dec: " << dur_.count() / float(nFrame) << std::endl;
		outputFile.close();
	};
};

static void __receiveHook(void* arg, uvg_rtp::frame::rtp_frame* frame)
{
	if (arg && frame)
	{
		static_cast<decoder*>(arg)->setLatestFrame(frame);
	}
}

class encoder {
public:
	encoder(CUdevice cudaDevice, std::atomic_bool& run):cudaDevice_(cudaDevice), run_(run) {

		sess = ctx.create_session("127.0.0.1");
		stream = sess->create_stream(8888, 8889, RTP_FORMAT_H264, RCE_NO_FLAGS);

		ck(cuCtxCreate(&cuContext_, 0, cudaDevice_));

		initializeParams.encodeConfig = &encodeConfig;		

		enc = std::make_unique<NvEncoderCuda>(cuContext_, w, h, NV_ENC_BUFFER_FORMAT_IYUV, 1);

		enc->CreateDefaultEncoderParams(
			&initializeParams,
			NV_ENC_CODEC_H264_GUID,
			NV_ENC_PRESET_P1_GUID, // what is this?
			NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY); // what does this mean?

		initializeParams.enablePTD = 1;
		initializeParams.enableSubFrameWrite = 1; // Speedup, explain why, maybe exclude that shit, maybe ref the FPGA macroblock pipeline or whatever
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
		enc->CreateEncoder(&initializeParams);

		encT_ = std::thread(&encoder::encProc, this);

	}

	~encoder() {
		if (encT_.joinable())
			encT_.join();

		ctx.destroy_session(sess);
	}

private:
	CUdevice cudaDevice_ = NULL;
	CUcontext cuContext_ = NULL;
	std::atomic_bool& run_;

	uvgrtp::context ctx;
	uvgrtp::session* sess = nullptr;
	uvgrtp::media_stream* stream = nullptr;

	std::unique_ptr<NvEncoderCuda> enc;
	std::thread encT_;

	NV_ENC_INITIALIZE_PARAMS initializeParams = { NV_ENC_INITIALIZE_PARAMS_VER };
	NV_ENC_CONFIG encodeConfig = { NV_ENC_CONFIG_VER };

	void encProc() {


		int counter = 0;
		std::chrono::milliseconds dur_ = std::chrono::milliseconds(0);

		try
		{
			int frameSize = enc->GetFrameSize();
			std::unique_ptr<uint8_t[]> pHostFrame(new uint8_t[frameSize]);

			std::streamsize nRead = 0;

			std::ifstream inputFile("C:/Users/vladimir/Desktop/vid.yuv", std::ifstream::in | std::ifstream::binary);
			if (!inputFile)
			{
				throw std::exception("can't open file");
			}

			do {
				auto start = std::chrono::high_resolution_clock::now();

				std::vector<std::vector<uint8_t>> vPacket;
				nRead = inputFile.read(reinterpret_cast<char*>(pHostFrame.get()), frameSize).gcount();

				if (nRead == frameSize)
				{
					const NvEncInputFrame* encoderInputFrame = enc->GetNextInputFrame();

					NvEncoderCuda::CopyToDeviceFrame(cuContext_, pHostFrame.get(), 0, (CUdeviceptr)encoderInputFrame->inputPtr,
						(int)encoderInputFrame->pitch,
						enc->GetEncodeWidth(),
						enc->GetEncodeHeight(),
						CU_MEMORYTYPE_HOST,
						encoderInputFrame->bufferFormat,
						encoderInputFrame->chromaOffsets,
						encoderInputFrame->numChromaPlanes);

					enc->EncodeFrame(vPacket);
				}
				else
				{
					enc->EndEncode(vPacket);
				}

				for (std::vector<uint8_t>& packet : vPacket) {

					// Allow time point to overflow, since it's symmetric on the other side
					//auto st = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch());
					//uint32_t ts = static_cast<uint32_t>(st.count());

					stream->push_frame(packet.data(), packet.size(), createTs(), RTP_NO_H26X_SCL);
				}


				// Finishing time.
				auto stop = std::chrono::high_resolution_clock::now();
				auto time = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
				dur_ += time;
				++counter;
				// Printing the measured time.
				//std::cout << "encoding: " << time.count() << " ms" << std::endl;
			} while (nRead == frameSize);

			enc->DestroyEncoder();
			inputFile.close();
		}
		catch (const std::exception& e)
		{
			std::cout << e.what() << std::endl;
		}

		run_ = false;

		auto resTime = std::chrono::duration_cast<std::chrono::milliseconds>(dur_);
		std::cout << "avg encoding: " << resTime.count() / float(counter) << " ms" << std::endl;
	
	};
};



int main()
{
	ck(cuInit(0));
	CUdevice cudaDevice = 0;
	ck(cuDeviceGet(&cudaDevice, 0));

	std::atomic_bool run = true;

	decoder dec(cudaDevice, run);
	encoder enc(cudaDevice, run);

}

