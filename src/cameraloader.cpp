#include "cameraloader.h"

#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/string_cast.hpp>

#include <boost/filesystem.hpp>
#include <boost/range/iterator_range.hpp>

#include <fstream>

namespace fs = boost::filesystem;

CameraLoader::CameraLoader(std::string poses_dir) {
    load(poses_dir);
}

void CameraLoader::load(std::string path) {
    if (fs::is_directory(path)) {
        for(auto& entry : boost::make_iterator_range(fs::directory_iterator(path), {})) {
            fs::path p_file(entry.path());
            int id = std::stoi(p_file.stem().string());
            std::cout << "id: " << id << ", File" << entry << "\n";
            loadPose(entry.path().string(), id);
        }
    }
    else {
        std::cout << "Error: " << path << " is not a directory.\n";
    }
}

void CameraLoader::loadPose(std::string path, int id) {
    int MAT_SIZE = 4;
    float data[MAT_SIZE * MAT_SIZE];
    std::ifstream infile(path);

    // Read pose line by line (4x4 matrix)
    for (int i = 0; i < MAT_SIZE * MAT_SIZE; i += MAT_SIZE) {
        infile >> data[i] >> data[i+1] >> data[i+2] >> data[i+3];

    }

    m_id_poses[id] = glm::make_mat4(data);
    std::cout << "Loaded pose: " << glm::to_string(m_id_poses[id]) << "\n";
}