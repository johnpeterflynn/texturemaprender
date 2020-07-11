#ifndef FRAME_WRITER_H
#define FRAME_WRITER_H

#include <boost/filesystem.hpp>

namespace fs = boost::filesystem;

class FrameWriter {
public:
    FrameWriter();
    FrameWriter(const std::string& output_path);

    void setPath(const std::string& output_path);

    void WriteAsTexcoord(const int id, const int height, const int width);
    void WriteAsJpg(const int id, const int height, const int width);

private:
    static void CompressWriteFile(char *buf, int size,
                                    const std::string& filename);

    fs::path m_output_path;
};

#endif // FRAME_WRITER_H
