#ifndef BASESCENE_H
#define BASESCENE_H

#include <glm/glm.hpp>

#include "shader_s.h"

class IScene {
public:
    virtual ~IScene() {}

    virtual glm::mat4 GetProjectionMatrix() = 0;
    virtual glm::mat4 GetViewMatrix() = 0;
    virtual glm::mat4 GetModelMatrix() = 0;

    virtual void Draw(Shader& shader) = 0;
};

#endif // BASESCENE_H
