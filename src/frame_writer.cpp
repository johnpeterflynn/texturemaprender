#include "frame_writer.h"

//#include <glad/glad.h>

#include <fstream>
#include <iostream>

#include <gzip/compress.hpp>
#include "stb/stb_image_write.h"
#include "utils.h"

FrameWriter::FrameWriter(int height, int width)
    : m_height(height)
    , m_width(width)
{
    int len_uv_data = m_height * m_width * NUM_UV_CHANNELS;
    m_data_raw.reserve(len_uv_data);
    m_data_significand.reserve(len_uv_data);
    
    // tell stb_image.h to flip loaded texture's on the y-axis (before loading model).
    //stbi_flip_vertically_on_write(false); // TODO: Why is this necessary?

    // TODO: Check if true: DNR might be flipped w.r.t. color/uv outputs
    stbi_flip_vertically_on_write(true);
}

FrameWriter::FrameWriter(int height, int width, const std::string& output_path)
    : FrameWriter(height, width)
{
   setPath(output_path);
}

FrameWriter::~FrameWriter() {
   // No reading or writing data vectors until last read finishes  
   if (m_p_write_thread) {
	   m_p_write_thread->join();
   } 
}

void FrameWriter::setPath(const std::string& output_path) {
   m_output_path = output_path;
}

void FrameWriter::ReadBufferAsTexcoord(std::vector<float> &uv_data) {
    glReadBuffer(GL_FRONT);
    glReadPixels(0, 0, m_width, m_height, GL_RGB, GL_FLOAT, uv_data.data());
}

void FrameWriter::RenderAsTexcoord(DNRenderer& dnr, bool writeout) {
    //std::string file_path = (m_output_path / std::to_string(id)).string();

    ReadBufferAsTexcoord(m_data_raw);

    // TODO: WARNING: dnr is given write access to data() which may be undesierable
    dnr.render(m_data_raw.data(), m_height, m_width, writeout);
}

void FrameWriter::WriteAsTexcoord(const int id) {
    std::string file_path = (m_output_path / std::to_string(id)).string();
    WriteAsTexcoord(file_path);
}

void FrameWriter::ConvertRawToSignificand(std::vector<unsigned short> &significand_out,
		const std::vector<float> &raw_in)
{
   // TODO: Do this in parallel
   for (int i = 0; i < m_height * m_width * NUM_UV_CHANNELS; i += NUM_UV_CHANNELS) {
   	for (int j = 0; j < NUM_UV_CHANNELS - 1; j++) {
      	    significand_out[i + j] = (unsigned short)(raw_in[i + j] * (1 << 16));
	}
	int mask_index = i + (NUM_UV_CHANNELS - 1);
	// TODO: WARNING: Mask conversion might suffer from rounding errors
      	significand_out[mask_index] = (unsigned short)(raw_in[mask_index] * 255.0);
   }
}

void FrameWriter::WriteAsTexcoord(const std::string& filename)
{
   ReadBufferAsTexcoord(m_data_raw);

   // No reading or writing data vectors until last read finishes  
   if (m_p_write_thread) {
	   m_p_write_thread->join();
   }
   
   ConvertRawToSignificand(m_data_significand, m_data_raw);   

   m_p_write_thread = std::make_shared<std::thread>(CompressWriteFile,
		   (char*)m_data_significand.data(),
                     m_height * m_width * NUM_UV_CHANNELS * sizeof(unsigned short),
                     filename);
}

void FrameWriter::WriteAsJpg(const std::string& filename) {
    GLchar data[m_height * m_width * NUM_COLOR_CHANNELS]; // # pixels x # floats per pixel
    std::string file_path;

    if (filename.empty()) {
        file_path = (m_output_path / dnr::time::getTimeAsString()).string();
    }
    else {
        // TODO: Writing of all objects should follow a consistent directory format
        //file_path = (m_output_path / filename).string();
        file_path = filename;
    }
    file_path += ".jpg";

    std::cout << "Writing snapshot to " << file_path << "\n";

    // TODO: BUG: Needed to switch this to GL_BACK to read the same image as WriteAsTexcoord().
    // Should they both be reading from the back buffer?
    glReadBuffer(GL_BACK);
    glReadPixels(0, 0, m_width, m_height, GL_RGB, GL_UNSIGNED_BYTE, data);

    // 90% quality, could be less
    stbi_write_jpg(file_path.c_str(), m_width, m_height, NUM_COLOR_CHANNELS, data, 100);
}

bool FrameWriter::SetupWriteVideo(float framerate) {
    std::string filename = dnr::time::getTimeAsString("recordings/") + ".avi";
    m_video_writer.open(filename,
                cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                framerate,
                cv::Size(m_width, m_height),
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
void FrameWriter::WriteFrameAsVideo() {
    cv::Mat pixels(m_height, m_width, CV_8UC3 );

    //use fast 4-byte alignment (default anyway) if possible
    glPixelStorei(GL_PACK_ALIGNMENT, (pixels.step & 3) ? 1 : 4);

    //set length of one complete row in destination data (doesn't need to equal img.cols)
    glPixelStorei(GL_PACK_ROW_LENGTH, pixels.step/pixels.elemSize());
    glReadBuffer(GL_FRONT);
    glReadPixels(0, 0, m_width, m_height, GL_RGB, GL_UNSIGNED_BYTE, pixels.data);

    cv::Mat cv_pixels(m_height, m_width, CV_8UC3 );
    for( int y=0; y<m_height; y++ ) for( int x=0; x<m_width; x++ )
    {
        cv_pixels.at<cv::Vec3b>(y,x)[2] = pixels.at<cv::Vec3b>(m_height-y-1,x)[0];
        cv_pixels.at<cv::Vec3b>(y,x)[1] = pixels.at<cv::Vec3b>(m_height-y-1,x)[1];
        cv_pixels.at<cv::Vec3b>(y,x)[0] = pixels.at<cv::Vec3b>(m_height-y-1,x)[2];
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
