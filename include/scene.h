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

    };
    Scene(const Params &params);

    /*virtual*/ glm::mat4 GetProjectionMatrix();
    /*virtual*/ glm::mat4 GetViewMatrix();
    /*virtual*/ glm::mat4 GetModelMatrix();

    virtual void Draw(Shader& shader);

   Camera m_camera;
   CameraLoader m_cam_loader;

private:

    void NotifyKeys(Key key, float deltaTime);
    void NotifyMouse(double xoffset, double yoffset);
    void NotifyScroll(double yoffset);

    SceneModel m_model;
    Model m_cube;
    std::shared_ptr<Model> m_submodel;
    int m_submodel_id;
    int m_num_submodules;

    float m_movement_speed;
    bool m_b_hold_object;
    float m_hold_object_dist;
};

#endif // SCENE_H
