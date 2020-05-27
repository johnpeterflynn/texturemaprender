#include "cameraloader.h"

#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/string_cast.hpp>

#include <boost/filesystem.hpp>
#include <boost/range/iterator_range.hpp>

#include <fstream>
#include <algorithm>

namespace fs = boost::filesystem;

CameraLoader::CameraLoader(std::string poses_dir) {
    load(poses_dir);
}

void CameraLoader::load(std::string path) {
    if (fs::is_directory(path)) {
        for(auto& entry : boost::make_iterator_range(fs::directory_iterator(path), {})) {
            fs::path p_file(entry.path());
            int id = std::stoi(p_file.stem().string());
            loadPose(entry.path().string(), id);
        }
    }
    else {
        std::cout << "Error: " << path << " is not a directory.\n";
    }

    std::cout << "Loaded " << m_ids.size() << " poses!" << "\n";

    // TODO: Create a datastructure automatically sorts data by key and
    // can be iterated over.
    std::sort(m_ids.begin(), m_ids.end());
}

void CameraLoader::loadPose(std::string path, int id) {
    int MAT_SIZE = 4;
    float data[MAT_SIZE * MAT_SIZE];
    std::ifstream infile(path);

    // Read pose line by line (4x4 matrix)
    for (int i = 0; i < MAT_SIZE * MAT_SIZE; i += MAT_SIZE) {
        infile >> data[i] >> data[i+1] >> data[i+2] >> data[i+3];

    }

    glm::mat4 m = glm::make_mat4(data);
    m = glm::transpose(m); // Entires are read row-wise but stored column-wise
    addPose(id, m);
}

void CameraLoader::addPose(int id, const glm::mat4& pose) {
    m_id_poses[id] = pose;
    m_ids.push_back(id);
}

int CameraLoader::getNumPoses() {
    return m_ids.size();
}

glm::mat4 CameraLoader::getPose(int index) {
    glm::mat4 pose = glm::mat4(1.0f);

    if (index < m_ids.size()) {
        int id = m_ids[index];
        pose = m_id_poses[id];
    }
    else {
        // TODO: Throw error
        std::cout << "Error: Attempted to access pose out of range.";
    }

    return pose;
}