#include "scene.h"

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/matrix_transform.hpp>


ModelDescriptor::ModelDescriptor(std::string name, std::string path, int mask)
    : m_name(name)
    , m_path(path)
    , m_b_loadable(true)
    , m_mask(mask)
{
}

ModelDescriptor::ModelDescriptor(std::string name, int id, int mask)
    : m_name(name)
    , m_id(id)
    , m_b_loadable(false)
    , m_mask(mask)
{
}

Scene::Scene(const Scene::Params &params)
    : m_default_params(params)
    , m_camera(glm::vec3(0.0f, 0.0f, 0.0f))
    , m_cam_loader(params.cam_params_dir, params.poses_dir)
    , m_model(params.model_path, params.aggregation_path, params.segs_path, params.scene_mask)
    , m_free_mode(m_default_params.free_mode)
    , m_movement_speed(2.5f)
    , m_b_hold_object(false)
    , m_first_update(true)
    , m_current_pose_id(-1)
    , m_pose_id_increment(1.0 / m_default_params.pose_interp_factor)
    , m_projection_mat(1.0f)
    , m_view_mat(1.0f)
    , m_model_mat(1.0f)
    , m_selected_library_model(-1)
    , m_selected_instantiated_model(-1)

{
    m_camera.setParams(m_cam_loader.m_intrinsics, m_cam_loader.m_extrinsics);

    auto model_semantic_labels = m_model.loadSegmentLabels();
    for (int i = 0; i < model_semantic_labels.size(); i++) {
        ModelDescriptor descriptor(model_semantic_labels[i], i, m_model.m_mask);
        m_model_library.push_back(descriptor);
    }
}


glm::mat4 Scene::GetProjectionMatrix(const float near, const float far) {
    return m_camera.GetProjectionMatrix(m_default_params.projection_height, m_default_params.projection_width, near, far);
}

glm::mat4 Scene::GetViewMatrix() {
    return m_view_mat;
}
/*
glm::mat4 Scene::GetModelMatrix() {
    return glm::mat4(1.0f);
}
*/

void Scene::Update() {
    // change state of whatever keeps track of the pose
    if (!m_free_mode) {
        if (m_first_update) {
            m_current_pose_id = 0.0;
            m_first_update = false;
        }
        else {
            m_current_pose_id += m_pose_id_increment;

        }
    }

    updateViewMatrix();
}

int Scene::GetCurrentPoseId() {
    // TODO: Find a way to better reflect the indices of the real poses. This blurs their definition
    // when pose_interp_factor != 1.0
    return m_current_pose_id * m_default_params.pose_interp_factor;
}

void Scene::updateViewMatrix() {
    // TODO: Resolve need to rotate the view by -90 and -180 degrees in these
    //  two modes.
    if (m_free_mode) {
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
    bool finished = !(m_current_pose_id + m_pose_id_increment < m_cam_loader.getNumPoses());
    return finished;
}

void Scene::Draw(Shader& shader) {
    glm::mat4 model_mat = glm::mat4(1.0);

    // TODO: Iterate over and draw each object (iterate from renderer?)
    model_mat = glm::translate(m_model.m_position);
    shader.setMat4("model", model_mat);
    m_model.Draw(shader);

    for (int i = 0; i < m_instantiated_models.size(); i++) {
        auto instance_model = m_instantiated_models[i];

        if (i == m_selected_instantiated_model) {
            if (m_b_hold_object) {
                instance_model->m_position = m_camera.m_position
                        + m_hold_object_dist * m_camera.Front;
            }
        }

        // TODO: Use a 4x4 matrix in object
        // TODO: Find the source of this weird issue where I need to rotate
        //  everything by 90 degrees
        // TODO: About what axis to define pitch and yaw, and in which order to
        //  apply rotations?
        model_mat = glm::rotate(glm::mat4(1.0f), glm::radians(90.0f),
                            glm::vec3(1.0f, 0.0f, 0.0f))
                *
                glm::translate(instance_model->m_position)
                * glm::scale(glm::vec3(1.0f, 1.0f, 1.0f) * instance_model->m_scale)
                * glm::rotate(instance_model->m_pitch, glm::vec3(0.0f, 0.0f, 1.0f))
                * glm::rotate(instance_model->m_yaw, glm::vec3(0.0f, 1.0f, 0.0f))
                * glm::rotate(glm::mat4(1.0f), glm::radians(-90.0f),
                              glm::vec3(1.0f, 0.0f, 0.0f));

        shader.setMat4("model", model_mat);
        instance_model->Draw(shader);
    }
}

void Scene::NotifyKeys(Key key, float deltaTime, bool is_already_pressed) {
    float velocity = m_movement_speed * deltaTime;

    std::shared_ptr<Model> selected_model = nullptr;
    if (m_instantiated_models.size() > 0) {
        selected_model = getSelectedInstanceModel();
    }

    // TODO: Optionally debounce the keys
    switch(key) {
     case Key::B:
        if (!is_already_pressed) {
            m_free_mode = !m_free_mode;
        }
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
        m_camera.ProcessKeyboard(TURN_UP, deltaTime);
        break;
     case Key::J:
        m_camera.ProcessKeyboard(TURN_LEFT, deltaTime);
        break;
     case Key::K:
        m_camera.ProcessKeyboard(TURN_DOWN, deltaTime);
        break;
     case Key::L:
        m_camera.ProcessKeyboard(TURN_RIGHT, deltaTime);
        break;
    case Key::Y:
        if (selected_model)
            selected_model->m_pitch += velocity;
        break;
    case Key::H:
        if (selected_model)
            selected_model->m_pitch -= velocity;
        break;
    case Key::T:
        if (selected_model)
            selected_model->m_yaw += velocity;
        break;
    case Key::G:
        if (selected_model)
            selected_model->m_yaw -= velocity;
        break;
    case Key::COMMA:
        if (selected_model)
            selected_model->m_scale -= velocity;
        break;
    case Key::PERIOD:
        if (selected_model)
            selected_model->m_scale += velocity;
        break;
    case Key::Q:
        if (!is_already_pressed) {
            m_model.deleteLabeledSubmodel(m_selected_library_model);
        }
        break;
    case Key::R:
        if (!is_already_pressed) {
            if (selected_model) {
                // Reset angles
                selected_model->m_pitch = 0;
                selected_model->m_yaw = 0;
            }
        }
        break;
    case Key::MINUS:
        if (!is_already_pressed) {
            setSelectedLibraryModel(std::max(0, m_selected_library_model - 1));
            std::cout << "Model " << m_selected_library_model << ": " << getSelectedLibraryModelDescriptor().name() << "\n";
        }
        break;
    case Key::EQUAL:
        if (!is_already_pressed) {
            setSelectedLibraryModel(std::min(m_selected_library_model + 1, int(m_model_library.size()) - 1));
            std::cout << "Model " << m_selected_library_model << ": " << getSelectedLibraryModelDescriptor().name() << "\n";
        }
        break;
    case Key::O:
        if (!is_already_pressed) {
            if (m_selected_instantiated_model >= 0) {
                if (selected_model && !m_b_hold_object) {
                    std::cout << "Grabbing object " << getSelectedInstanceModelDescriptor().name()
                              << " " << m_selected_instantiated_model << "\n";
                    m_hold_object_dist = glm::distance(m_camera.m_position, selected_model->m_position);
                    m_b_hold_object = true;
                }
                else {
                    std::cout << "Releasing object " << getSelectedInstanceModelDescriptor().name()
                              << " " << m_selected_instantiated_model << "\n";
                    m_b_hold_object = false;
                }
            }
            else {
                std::cout << "WARNING: There are no instantiated objects " << "\n";
            }
        }
        break;
    case Key::M:
        if (!is_already_pressed) {
            // Get descriptor for currently selected model
            ModelDescriptor desc = getSelectedLibraryModelDescriptor();
            std::shared_ptr<Model> new_model;
            if (!desc.m_b_loadable) {
                // Create that model
                new_model = m_model.extractLabeledSubmodel(m_selected_library_model);
            }
            else {
                new_model = std::make_shared<Model>(desc.path(), false, desc.mask());
            }
            // Add new model
            addInstanceModel(new_model, desc);
            // Set focus to that instantiated model
            setSelectedInstanceModel(m_instantiated_models.size() - 1);
        }
        break;
    case Key::N:
        if (!is_already_pressed) {
            if(m_selected_instantiated_model >= 0) {
                deleteInstanceModel(m_selected_instantiated_model);

                // Implicitly sets m_selected_instantiated_model to -1 when no more instance models exist
                int new_id = min(m_selected_instantiated_model, int(m_instantiated_models.size()) - 1);
                setSelectedInstanceModel(new_id);
            }
            else {
                std::cout << "WARNING: There are no instantiated objects " << "\n";
            }
        }
        break;
    case Key::NINE:
        if (!is_already_pressed) {
            if (m_selected_instantiated_model >= 0) {
                setSelectedInstanceModel(std::max(0, m_selected_instantiated_model - 1));
                std::cout << "Model " << m_selected_instantiated_model << ": " << getSelectedInstanceModelDescriptor().name() << "\n";
            }
            else {
                std::cout << "WARNING: There are no instantiated objects " << "\n";
            }
        }
        break;
    case Key::ZERO:
        if (!is_already_pressed) {
            if (m_selected_instantiated_model >= 0) {
                setSelectedInstanceModel(std::min(m_selected_instantiated_model + 1, int(m_instantiated_models.size()) - 1));
                std::cout << "Model " << m_selected_instantiated_model << ": " << getSelectedInstanceModelDescriptor().name() << "\n";
            }
            else {
                std::cout << "WARNING: There are no instantiated objects " << "\n";
            }
        }
        break;
    }
}

ModelDescriptor Scene::getSelectedLibraryModelDescriptor() {
    return m_model_library[m_selected_library_model];
}

void Scene::setSelectedLibraryModel(int id) {
    m_selected_library_model = id;
}

ModelDescriptor Scene::getSelectedInstanceModelDescriptor() {
    if (m_selected_instantiated_model >= m_instantiated_models.size()) {
        std::cout << "ERROR: Attempting to select descriptor for instance model " << m_selected_instantiated_model << " which is out of range\n";
    }
    return m_instantiated_model_descriptors[m_selected_instantiated_model];
}

std::shared_ptr<Model> Scene::getSelectedInstanceModel() {
    if (m_selected_instantiated_model >= m_instantiated_models.size()) {
        std::cout << "ERROR: Attempting to select instance model " << m_selected_instantiated_model << " which is out of range\n";
        return nullptr;
    }
    return m_instantiated_models[m_selected_instantiated_model];
}

void Scene::setSelectedInstanceModel(int id) {
    m_selected_instantiated_model = id;
}

void Scene::addInstanceModel(std::shared_ptr<Model> model, ModelDescriptor desc) {
    // Add model to list of instantiated models
    m_instantiated_models.push_back(model);
    // Add model descriptor to list of instantiated descriptors
    m_instantiated_model_descriptors.push_back(desc);
}

void Scene::deleteInstanceModel(int index) {
    m_instantiated_models.erase(m_instantiated_models.begin()+index);
    m_instantiated_model_descriptors.erase(m_instantiated_model_descriptors.begin()+index);
}

void Scene::NotifyMouse(double xoffset, double yoffset)
{
    m_camera.ProcessMouseMovement(xoffset, yoffset);
}

void Scene::NotifyScroll(double yoffset)
{
    m_camera.ProcessMouseScroll(yoffset);
}
