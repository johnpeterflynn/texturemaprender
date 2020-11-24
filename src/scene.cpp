#include "scene.h"

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/matrix_transform.hpp>

Scene::Scene(const Scene::Params &params)
    : m_camera(glm::vec3(0.0f, 0.0f, 0.0f))
    , m_cam_loader(params.cam_params_dir, params.poses_dir)
    , m_model(params.model_path, params.aggregation_path, params.segs_path, params.scene_mask)
    , m_movement_speed(2.5f)
    , m_b_hold_object(false)
    , m_submodel(nullptr)
    , m_submodel_id(0)
    , m_num_submodules(67) // TODO: Set this from segs file

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
    glm::mat4 model = glm::mat4(1.0);

    // TODO: Iterate over and draw each object (iterate from renderer?)
    model = glm::translate(m_model.m_position);
    shader.setMat4("model", model);
    m_model.Draw(shader);

    if (m_b_hold_object) {
        m_submodel->m_position = m_camera.m_position
                + m_hold_object_dist * m_camera.Front;
    }

    if (m_submodel) {
        // TODO: Use a 4x4 matrix in object
        // TODO: Find the source of this weird issue where I need to rotate
        //  everything by 90 degrees
        // TODO: About what axis to define pitch and yaw, and in which order to
        //  apply rotations?
        model = glm::rotate(glm::mat4(1.0f), glm::radians(90.0f),
                            glm::vec3(1.0f, 0.0f, 0.0f))
                *
                glm::translate(m_submodel->m_position)
                * glm::rotate(m_submodel->m_pitch, glm::vec3(0.0f, 0.0f, 1.0f))
                * glm::rotate(m_submodel->m_yaw, glm::vec3(0.0f, 1.0f, 0.0f))
                * glm::rotate(glm::mat4(1.0f), glm::radians(-90.0f),
                              glm::vec3(1.0f, 0.0f, 0.0f));
        shader.setMat4("model", model);
        m_submodel->Draw(shader);
    }
}

void Scene::NotifyKeys(Key key, float deltaTime) {
    float velocity = m_movement_speed * deltaTime;

    // TODO: Optionally debounce the keys
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
     case Key::I:
        m_submodel->m_position.x += velocity;
        break;
     case Key::J:
        m_submodel->m_position.y += velocity;
        break;
     case Key::K:
        m_submodel->m_position.x -= velocity;
        break;
     case Key::L:
        m_submodel->m_position.y -= velocity;
        break;
    case Key::Y:
        m_submodel->m_pitch += velocity;
        break;
    case Key::H:
        m_submodel->m_pitch -= velocity;
        break;
    case Key::T:
        m_submodel->m_yaw += velocity;
        break;
    case Key::G:
        m_submodel->m_yaw -= velocity;
        break;
    case Key::R:
        // Reset angles
        m_submodel->m_pitch = 0;
        m_submodel->m_yaw = 0;
        break;
    case Key::MINUS:
        m_submodel_id = std::max(0, m_submodel_id - 1);
        std::cout << "ID: " << m_submodel_id << "\n";
        break;
    case Key::EQUAL:
        m_submodel_id = std::min(m_submodel_id + 1, m_num_submodules - 1);
        std::cout << "ID: " << m_submodel_id << "\n";
        break;
    case Key::O:
        if (!m_b_hold_object) {
            std::cout << "Grabbing object " << m_submodel_id << "\n";
            m_hold_object_dist = glm::distance(m_camera.m_position, m_submodel->m_position);
            m_b_hold_object = true;
        }
        else {
            std::cout << "Releasing object " << m_submodel_id << "\n";
            m_b_hold_object = false;
        }
        break;
    case Key::M:
        m_submodel = m_model.extractLabeledSubmodel(m_submodel_id);
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
