#ifndef SCENE_H
#define SCENE_H

//#include "interfaces/iscene.h"
#include "listeners/keylistener.h"
#include "camera.h"
#include "cameraloader.h"
#include "scenemodel.h"

class Scene : /*public IScene,*/ public KeyListener {
public:
    struct Params {
        std::string model_path;
        std::string aggregation_path;
        std::string segs_path;
        std::string cam_params_dir;
        std::string poses_dir;
        int scene_mask;
        int projection_height;
        int projection_width;

    };
    Scene(const Params &params);

    /*virtual*/ glm::mat4 GetProjectionMatrix(const float near, const float far);
    /*virtual*/ glm::mat4 GetViewMatrix();
    /*virtual*/ glm::mat4 GetModelMatrix();
    int GetCurrentPoseId();

    void Update(bool free_mode);
    virtual void Draw(Shader& shader);

    bool isFinished();

   Camera m_camera;
   CameraLoader m_cam_loader;

private:

    void NotifyKeys(Key key, float deltaTime);
    void NotifyMouse(double xoffset, double yoffset);
    void NotifyScroll(double yoffset);

    void updateViewMatrix(bool free_mode) ;

    Params m_params;

    SceneModel m_model;
    Model m_cube;
    std::shared_ptr<Model> m_submodel;
    int m_submodel_id;
    int m_num_submodules;

    float m_movement_speed;
    bool m_b_hold_object;
    float m_hold_object_dist;

    bool m_first_update; // Allow setup functionality on the first call to Update()
    float m_current_pose_id; // Currently only relevant when free_mode == false
    float m_pose_interp_factor;

    glm::mat4 m_projection_mat;
    glm::mat4 m_view_mat;
    glm::mat4 m_model_mat;
};

#endif // SCENE_H
