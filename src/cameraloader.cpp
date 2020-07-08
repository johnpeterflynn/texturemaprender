#include "cameraloader.h"

#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/string_cast.hpp>

#include <boost/filesystem.hpp>
#include <boost/range/iterator_range.hpp>

#include <fstream>
#include <algorithm>

namespace fs = boost::filesystem;

CameraLoader::CameraLoader(std::string params_dir, std::string poses_dir) {
    loadParams(params_dir);
    loadPoses(poses_dir);
}

void CameraLoader::loadParams(std::string params_dir) {
    fs::path intrinsics_path = fs::path(params_dir) / "intrinsic_color.txt";
    fs::path extrinsics_path = fs::path(params_dir) / "extrinsic_color.txt";

    m_intrinsics = loadMat4(intrinsics_path.string());
    m_extrinsics = loadMat4(extrinsics_path.string());
}

void CameraLoader::loadPoses(std::string path) {
    if (fs::is_directory(path)) {
        for(auto& entry : boost::make_iterator_range(fs::directory_iterator(path), {})) {
            fs::path p_file(entry.path());
            int id = std::stoi(p_file.stem().string());
            auto pose = loadMat4(entry.path().string());
            addPose(id, pose);
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

glm::mat4 CameraLoader::loadMat4(std::string path) {
    int MAT_SIZE = 4;
    float data[MAT_SIZE * MAT_SIZE];
    std::ifstream infile(path);

    // Read pose line by line (4x4 matrix)
    for (int i = 0; i < MAT_SIZE * MAT_SIZE; i += MAT_SIZE) {
        infile >> data[i] >> data[i+1] >> data[i+2] >> data[i+3];
    }

    glm::mat4 m = glm::make_mat4(data);
    m = glm::transpose(m); // Entires are read row-wise but stored column-wise

    return m;
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