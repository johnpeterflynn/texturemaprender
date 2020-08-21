#ifndef SCENE_H
#define SCENE_H

//#include "interfaces/iscene.h"
#include "listeners/keylistener.h"
#include "camera.h"
#include "cameraloader.h"
#include "model.h"

class Scene : /*public IScene,*/ public KeyListener {
public:
    Scene(std::string const &model_path, std::string const &cam_params_dir,
          std::string const &poses_dir);

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

    Model m_model;
};

#endif // SCENE_H
