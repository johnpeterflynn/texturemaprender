#include "scene.h"

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/matrix_transform.hpp>

Scene::Scene(const Scene::Params &params)
    : m_params(params)
    , m_camera(glm::vec3(0.0f, 0.0f, 0.0f))
    , m_cam_loader(params.cam_params_dir, params.poses_dir)
    , m_model(params.model_path, params.aggregation_path, params.segs_path, params.scene_mask)
    , m_movement_speed(2.5f)
    , m_b_hold_object(false)
    , m_submodel(nullptr)
    , m_submodel_id(0)
    , m_num_submodules(67) // TODO: Set this from segs file
    , m_first_update(true)
    , m_current_pose_id(-1)
    , m_pose_id_increment(1.0 / m_params.pose_interp_factor)
    , m_projection_mat(1.0f)
    , m_view_mat(1.0f)
    , m_model_mat(1.0f)

{
    m_camera.setParams(m_cam_loader.m_intrinsics, m_cam_loader.m_extrinsics);
}


glm::mat4 Scene::GetProjectionMatrix(const float near, const float far) {
    return m_camera.GetProjectionMatrix(m_params.projection_height, m_params.projection_width, near, far);
}

glm::mat4 Scene::GetViewMatrix() {
    return m_view_mat;
}
/*
glm::mat4 Scene::GetModelMatrix() {
    return glm::mat4(1.0f);
}
*/

void Scene::Update(bool free_mode) {
    // change state of whatever keeps track of the pose
    if (!free_mode) {
        if (m_first_update) {
            m_current_pose_id = 0.0;
            m_first_update = false;
        }
        else {
            m_current_pose_id += m_pose_id_increment;

        }
    }

    updateViewMatrix(free_mode);
}

int Scene::GetCurrentPoseId() {
    // TODO: Find a way to better reflect the indices of the real poses. This blurs their definition
    // when pose_interp_factor != 1.0
    return m_current_pose_id * m_params.pose_interp_factor;
}

void Scene::updateViewMatrix(bool free_mode) {
    // TODO: Resolve need to rotate the view by -90 and -180 degrees in these
    //  two modes.
    if (free_mode) {
        // TODO: Make m_camera private
        m_view_mat = m_camera.GetViewMatrix()
                * glm::rotate(glm::mat4(1.0f), glm::radians(-90.0f),
                              glm::vec3(1.0f, 0.0f, 0.0f));
    }
    else {
        // TODO: Make m_camera_loader private
        // Rotate to make +Z the up direction as often defined by 3D scans
        m_view_mat = glm::rotate(glm::mat4(1.0f), glm::radians(-180.0f),
                           glm::vec3(1.0f, 0.0f, 0.0f))
                * glm::inverse(m_cam_loader.getInterpolatedPose(m_current_pose_id));
    }
}

bool Scene::isFinished() {
    // True when the final pose_id is reached. Currently defined as the the final poses
    // in a numbered sequence of poses. Note that we finish once m_current_pose_id is one
    // less than the number of poses since the referenced id is processed as soon as it
    // is set (in Update()).
    return !(m_current_pose_id + m_pose_id_increment < (m_cam_loader.getNumPoses() - 1));
}

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
