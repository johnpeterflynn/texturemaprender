#ifndef OBJECT_H
#define OBJECT_H

#include <glm/glm.hpp>

class Object {
public:
    Object();
    Object(glm::vec3 position, float yaw, float pitch);

    // Position
    glm::vec3 m_position;

    // Orientation
    float m_yaw;
    float m_pitch;
};

#endif // OBJECT_H
