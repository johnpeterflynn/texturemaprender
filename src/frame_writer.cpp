#include "frame_writer.h"

#include <glad/glad.h>

#include <fstream>
#include <iostream>

#include <gzip/compress.hpp>
#include "stb_image_write.h"

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
    int channels = 3;
    // TODO: Allocate and deallocate heap_data only once
    float *heap_data = new float[height * width * channels];
    //std::string file_path = (m_output_path / std::to_string(id)).string();

    glReadBuffer(GL_FRONT);
    glReadPixels(0, 0, width, height, GL_RGB, GL_FLOAT, heap_data);

    dnr.render(heap_data, height, width, writeout);

    delete [] heap_data;
}

void FrameWriter::WriteAsTexcoord(const int id, const int height, const int width)
{
   int channels = 3;
   // TODO: Allocate and deallocate heap_data only once
   float *heap_data = new float[height * width * channels];
   std::string file_path = (m_output_path / std::to_string(id)).string();

   glReadBuffer(GL_FRONT);
   glReadPixels(0, 0, width, height, GL_RGB, GL_FLOAT, heap_data);

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

void FrameWriter::CompressWriteFile(char *buf, int size,
                                      const std::string& filename)
{
    std::string compressed_data = gzip::compress(buf, size);

    auto myfile = std::fstream(filename + ".gz", std::ios::out | std::ios::binary);
    myfile.write(compressed_data.data(), compressed_data.size());
    myfile.close();

}
