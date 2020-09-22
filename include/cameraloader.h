#ifndef CAMERALOADER_H
#define CAMERALOADER_H

#include <glm/glm.hpp>

#include <iostream>
#include <map>
#include <vector>
#include <string>

class CameraLoader {
public:
    CameraLoader(std::string params_dir, std::string poses_dir);

    void addPose(int id, const glm::mat4& pose);
    int getNumPoses();
    glm::mat4 getPose(int index);
    void savePose(const glm::mat4& mat, const std::string& path);

    glm::mat4 m_intrinsics;
    glm::mat4 m_extrinsics;

private:
    void loadParams(std::string path);
    void loadPoses(std::string path);
    glm::mat4 loadMat4(std::string path);
    void saveMat4(const glm::mat4& mat, const std::string& path);

    std::vector<int> m_ids;
    std::map<int, glm::mat4> m_id_poses;
};

#endif // CAMERALOADER_H
