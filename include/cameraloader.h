#ifndef CAMERALOADER_H
#define CAMERALOADER_H

#include <glm/glm.hpp>

#include <iostream>
#include <map>
#include <string>

class CameraLoader {
public:
    void load(std::string path);
    void loadPose(std::string path, int id);

private:
    std::map<int, glm::mat4> m_id_poses;
};

#endif // CAMERALOADER_H
