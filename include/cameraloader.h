#ifndef CAMERALOADER_H
#define CAMERALOADER_H

#include <glm/glm.hpp>

#include <iostream>
#include <map>
#include <vector>
#include <string>

class CameraLoader {
public:
    CameraLoader(std::string poses_dir);

    void addPose(int id, const glm::mat4& pose);
    int getNumPoses();
    glm::mat4 getPose(int index);

private:
    void load(std::string path);
    void loadPose(std::string path, int id);

private:
    std::vector<int> m_ids;
    std::map<int, glm::mat4> m_id_poses;
};

#endif // CAMERALOADER_H
