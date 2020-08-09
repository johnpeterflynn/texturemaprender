#include "scene.h"

Scene::Scene(std::string const &model_path)
    : m_model(model_path)
{

}

glm::mat4 Scene::GetProjectionMatrix() {
    return glm::mat4(1.0f);
}

glm::mat4 Scene::GetViewMatrix() {
    return glm::mat4(1.0f);
}

glm::mat4 Scene::GetModelMatrix() {
    return glm::mat4(1.0f);
}

void Scene::Draw(Shader& shader) {
    m_model.Draw(shader);
}