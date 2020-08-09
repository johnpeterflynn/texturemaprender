#ifndef SCENE_H
#define SCENE_H

#include "interfaces/iscene.h"
#include "model.h"

class Scene : public IScene {
public:
    Scene(std::string const &model_path);

    virtual glm::mat4 GetProjectionMatrix();
    virtual glm::mat4 GetViewMatrix();
    virtual glm::mat4 GetModelMatrix();

    virtual void Draw(Shader& shader);
private:
    Model m_model;
};

#endif // SCENE_H
