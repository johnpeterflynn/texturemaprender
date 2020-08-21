#include "scene.h"

#include <glm/glm.hpp>

Scene::Scene(std::string const &model_path,
             std::string const &cam_params_dir,
             std::string const &poses_dir)
    : m_camera(glm::vec3(0.0f, 0.0f, 0.0f))
    , m_cam_loader(cam_params_dir, poses_dir)
    , m_model(model_path)
{
    m_camera.setParams(m_cam_loader.m_intrinsics, m_cam_loader.m_extrinsics);
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
        //frameWriter.WriteAsJpg(num_snapshots++, SCR_HEIGHT, SCR_WIDTH);
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