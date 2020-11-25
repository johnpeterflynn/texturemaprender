#include "object.h"

Object::Object()
    : Object(glm::vec3(0.0f, 0.0f, 0.0f), 0.0, 0.0)
{
}

Object::Object(glm::vec3 position, float yaw, float pitch)
    : m_position(position)
    , m_yaw(yaw)
    , m_pitch(pitch)
    , m_scale(1.0f)
{

}
