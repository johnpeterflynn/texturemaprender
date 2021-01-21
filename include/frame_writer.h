#ifndef FRAME_WRITER_H
#define FRAME_WRITER_H

#include <boost/filesystem.hpp>

// TODO: Do not include the entire opencv library
//#include <opencv2/core.hpp>     // Basic OpenCV structures (cv::Mat)
//#include <opencv2/videoio.hpp>  // Video write
#include <opencv4/opencv2/opencv.hpp>
//#include <opencv/cv.hpp>
#include <thread>

#include "deferred_neural_renderer.h"

namespace fs = boost::filesystem;

class FrameWriter {
public:
    FrameWriter(int height, int width);
    FrameWriter(int height, int width, const std::string& output_path);

    void setPath(const std::string& output_path);

    void RenderAsTexcoord(DNRenderer& dnr, bool writeout);

    void WriteAsTexcoord(const int id);
    void WriteAsTexcoord(const std::string& filename = "");
    void WriteAsJpg(const std::string& filename = "");

    bool SetupWriteVideo(float framerate = 15.0f);
    void ShutdownWriteVideo();
    bool WriteVideoReady();
    void WriteFrameAsVideo();

private:
    void ReadBufferAsTexcoord(std::vector<float> &data);
    void ConvertRawToSignificand(std::vector<unsigned short> &significand_out,
		const std::vector<float> &raw_in);
    static void CompressWriteFile(char *buf, int size,
                                    const std::string& filename);

    static const unsigned int NUM_UV_CHANNELS = 3;
    static const unsigned int NUM_COLOR_CHANNELS = 3;

    fs::path m_output_path;
    cv::VideoWriter m_video_writer;

    int m_height;
    int m_width;

    std::vector<float> m_data_raw;
    std::vector<unsigned short> m_data_significand;
    std::shared_ptr<std::thread> m_p_write_thread;
};

#endif // FRAME_WRITER_H
