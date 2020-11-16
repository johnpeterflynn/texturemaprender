#ifndef FRAME_WRITER_H
#define FRAME_WRITER_H

#include <boost/filesystem.hpp>

// TODO: Do not include the entire opencv library
//#include <opencv2/core.hpp>     // Basic OpenCV structures (cv::Mat)
//#include <opencv2/videoio.hpp>  // Video write
#include <opencv4/opencv2/opencv.hpp>
//#include <opencv/cv.hpp>

#include "deferred_neural_renderer.h"

namespace fs = boost::filesystem;

class FrameWriter {
public:
    FrameWriter();
    FrameWriter(const std::string& output_path);

    void setPath(const std::string& output_path);

    void RenderAsTexcoord(DNRenderer& dnr, int rows, int cols, bool writeout);

    void WriteAsTexcoord(const int height, const int width, const int id);
    void WriteAsTexcoord(const int height, const int width, const std::string& filename = "");
    void WriteAsJpg(const int height, const int width, const std::string& filename = "");

    bool SetupWriteVideo(int height, int width, float framerate = 15.0f);
    void ShutdownWriteVideo();
    bool WriteVideoReady();
    void WriteFrameAsVideo(int height, int width);

private:
    static void CompressWriteFile(char *buf, int size,
                                    const std::string& filename);

    fs::path m_output_path;
    cv::VideoWriter m_video_writer;
};

#endif // FRAME_WRITER_H
