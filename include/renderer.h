#ifndef RENDERER_H
#define RENDERER_H

#include "camera.h"
#include "deferred_neural_renderer.h"
#include "frame_writer.h"
//#include "interfaces/iscene.h"
#include "listeners/keylistener.h"
#include "scene.h"
#include "shader_s.h"

class Renderer : public KeyListener {
public:
    Renderer(int height, int width, const std::string &net_path,
             const std::string &output_path);

    // TODO: Make params const
    void Draw(Scene& scene, int pose_id, bool free_mode, bool writeToFile);

private:
    void NotifyKeys(Key key, float deltaTime);

    const int m_height, m_width;

    Shader m_uv_shader;
    Shader m_color_shader;
    Shader m_texture_shader;

    FrameWriter m_frameWriter;

    DNRenderer m_dnr;

    unsigned int m_framebuffer;
    unsigned int m_texColorBuffer;
    unsigned int m_renderbuffer;
    unsigned int m_quadVAO, m_quadVBO;

    bool m_b_snapshot;
};

#endif // RENDERER_H
