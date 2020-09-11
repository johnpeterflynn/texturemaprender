#include "frame_writer.h"

#include <glad/glad.h>

#include <fstream>
#include <iostream>

#include <gzip/compress.hpp>
#include "stb/stb_image_write.h"
#include "utils.h"

FrameWriter::FrameWriter() {
    // tell stb_image.h to flip loaded texture's on the y-axis (before loading model).
    stbi_flip_vertically_on_write(false); // TODO: Why is this necessary?
}

FrameWriter::FrameWriter(const std::string& output_path)
    : FrameWriter()
{
   setPath(output_path);
}

void FrameWriter::setPath(const std::string& output_path) {
   m_output_path = output_path;
}

void FrameWriter::RenderAsTexcoord(DNRenderer& dnr, int height, int width, bool writeout) {
    int channels = 2;
    // TODO: Allocate and deallocate heap_data only once
    float *heap_data = new float[height * width * channels];
    //std::string file_path = (m_output_path / std::to_string(id)).string();

    glReadBuffer(GL_FRONT);
    glReadPixels(0, 0, width, height, GL_RG, GL_FLOAT, heap_data);

    dnr.render(heap_data, height, width, writeout);

    delete [] heap_data;
}

void FrameWriter::WriteAsTexcoord(const int id, const int height, const int width)
{
   int channels = 2;
   // TODO: Allocate and deallocate heap_data only once
   float *heap_data = new float[height * width * channels];
   std::string file_path = (m_output_path / std::to_string(id)).string();

   glReadBuffer(GL_FRONT);
   glReadPixels(0, 0, width, height, GL_RG, GL_FLOAT, heap_data);

   CompressWriteFile((char*)heap_data,
                     height * width * channels * sizeof(float),
                     file_path);

   delete [] heap_data;
}

void FrameWriter::WriteAsJpg(const int id, const int height, const int width) {
    int num_jpg_channels = 3;
    GLchar data[height * width * num_jpg_channels]; // # pixels x # floats per pixel
    std::string file_path = (m_output_path / std::to_string(id)).string() + ".jpg";

    glReadBuffer(GL_FRONT);
    glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, data);

    // 90% quality, could be less
    stbi_write_jpg(file_path.c_str(), width, height, num_jpg_channels, data, 90);
}

bool FrameWriter::SetupWriteVideo(int height, int width, float framerate) {
    std::string filename = dnr::time::getTimeAsString("recordings/") + ".avi";
    m_video_writer.open(filename,
                cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                framerate,
                cv::Size(width, height),
                true);

    bool b_opened = m_video_writer.isOpened();

    if (b_opened) {
        std::cout  << "Recording video to: " << filename << "\n";
    }
    else {
        std::cout  << "Could not create video file for writing: " << filename << "\n";
    }

    return b_opened;
}

void FrameWriter::ShutdownWriteVideo() {
    m_video_writer.release();
}

bool FrameWriter::WriteVideoReady() {
    return m_video_writer.isOpened();
}

// TODO: Save height and width from setup
// TODO: Record in a background thread
void FrameWriter::WriteFrameAsVideo(int height, int width) {
    cv::Mat pixels(height, width, CV_8UC3 );

    glReadBuffer(GL_FRONT);
    glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, pixels.data);

    cv::Mat cv_pixels(height, width, CV_8UC3 );
    for( int y=0; y<height; y++ ) for( int x=0; x<width; x++ )
    {
        cv_pixels.at<cv::Vec3b>(y,x)[2] = pixels.at<cv::Vec3b>(height-y-1,x)[0];
        cv_pixels.at<cv::Vec3b>(y,x)[1] = pixels.at<cv::Vec3b>(height-y-1,x)[1];
        cv_pixels.at<cv::Vec3b>(y,x)[0] = pixels.at<cv::Vec3b>(height-y-1,x)[2];
    }

    // Write next frame
    m_video_writer.write(cv_pixels);
}

void FrameWriter::CompressWriteFile(char *buf, int size,
                                      const std::string& filename)
{
    std::string compressed_data = gzip::compress(buf, size);

    auto myfile = std::fstream(filename + ".gz", std::ios::out | std::ios::binary);
    myfile.write(compressed_data.data(), compressed_data.size());
    myfile.close();

}
