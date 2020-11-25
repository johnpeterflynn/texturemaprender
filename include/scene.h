#ifndef SCENE_H
#define SCENE_H

//#include "interfaces/iscene.h"
#include "listeners/keylistener.h"
#include "camera.h"
#include "cameraloader.h"
#include "scenemodel.h"

class ModelDescriptor {
public:
    ModelDescriptor(std::string name, std::string path);
    ModelDescriptor(std::string name, int id);

    std::string m_name;
    std::string m_path;
    int m_id;
    bool m_b_loadable;

    std::string name() {
        return m_name;
    }

    std::string path() {
        return m_path;
    }

    int id() {
        return m_id;
    }
};

class Scene : /*public IScene,*/ public KeyListener {
public:
    struct Params {
        std::string model_path;
        std::string aggregation_path;
        std::string segs_path;
        std::string cam_params_dir;
        std::string poses_dir;
        bool free_mode;
        int scene_mask;
        int projection_height;
        int projection_width;
        float pose_interp_factor;

    };
    Scene(const Params &params);

    /*virtual*/ glm::mat4 GetProjectionMatrix(const float near, const float far);
    /*virtual*/ glm::mat4 GetViewMatrix();
    /*virtual*/ glm::mat4 GetModelMatrix();
    int GetCurrentPoseId();

    void Update();
    virtual void Draw(Shader& shader);

    bool isFinished();

    ModelDescriptor getSelectedLibraryModelDescriptor();
    void setSelectedLibraryModel(int id);
    ModelDescriptor getSelectedInstanceModelDescriptor();
    std::shared_ptr<Model> getSelectedInstanceModel();
    void setSelectedInstanceModel(int id);
    void addInstanceModel(std::shared_ptr<Model> model, ModelDescriptor desc);

   Camera m_camera;
   CameraLoader m_cam_loader;

private:

    void NotifyKeys(Key key, float deltaTime, bool is_already_pressed);
    void NotifyMouse(double xoffset, double yoffset);
    void NotifyScroll(double yoffset);

    void updateViewMatrix();

    const Params m_default_params;

    SceneModel m_model;
    Model m_cube;

    bool m_free_mode; // True if user can control the camera

    float m_movement_speed;
    bool m_b_hold_object;
    float m_hold_object_dist;

    bool m_first_update; // Allow setup functionality on the first call to Update()
    float m_current_pose_id; // Currently only relevant when free_mode == false
    float m_pose_id_increment; // Amount that pose id increases on each Update()

    glm::mat4 m_projection_mat;
    glm::mat4 m_view_mat;
    glm::mat4 m_model_mat;

    int m_selected_library_model;
    int m_selected_instantiated_model;
    std::vector<ModelDescriptor> m_model_library;
    std::vector<std::shared_ptr<Model>> m_instantiated_models;
    std::vector<ModelDescriptor> m_instantiated_model_descriptors;
};

#endif // SCENE_H
