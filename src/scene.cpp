#include "scene.h"

#include <glm/glm.hpp>

Scene::Scene(const Scene::Params &params)
    : m_camera(glm::vec3(0.0f, 0.0f, 0.0f))
    , m_cam_loader(params.cam_params_dir, params.poses_dir)
    , m_model(params.model_path, params.aggregation_path, params.segs_path)
{
    m_camera.setParams(m_cam_loader.m_intrinsics, m_cam_loader.m_extrinsics);
}

/*
glm::mat4 Scene::GetProjectionMatrix() {
    return glm::mat4(1.0f);
}

glm::mat4 Scene::GetViewMatrix() {
    return glm::mat4(1.0f);
}

glm::mat4 Scene::GetModelMatrix() {
    return glm::mat4(1.0f);
}
*/

void Scene::Draw(Shader& shader) {
    m_model.Draw(shader);
}

void Scene::NotifyKeys(Key key, float deltaTime) {
    switch(key) {
     case Key::W:
        m_camera.ProcessKeyboard(FORWARD, deltaTime);
        break;
     case Key::S:
        m_camera.ProcessKeyboard(BACKWARD, deltaTime);
        break;
     case Key::A:
        m_camera.ProcessKeyboard(LEFT, deltaTime);
        break;
     case Key::D:
        m_camera.ProcessKeyboard(RIGHT, deltaTime);
        break;
     case Key::SPACE:
        m_camera.ProcessKeyboard(UP, deltaTime);
        break;
     case Key::LEFT_SHIFT:
        m_camera.ProcessKeyboard(DOWN, deltaTime);
        break;
    }
}

void Scene::NotifyMouse(double xoffset, double yoffset)
{
    m_camera.ProcessMouseMovement(xoffset, yoffset);
}

void Scene::NotifyScroll(double yoffset)
{
    m_camera.ProcessMouseScroll(yoffset);
}