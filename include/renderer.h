#ifndef RENDERER_H
#define RENDERER_H

#include "camera.h"
#include "deferred_neural_renderer.h"
#include "frame_writer.h"
#include "interfaces/iscene.h"
#include "shader_s.h"

class Renderer {
public:
    Renderer(int height, int width, const std::string &net_path,
             const std::string &output_path);

    // TODO: Make params const
    void Draw(IScene& scene, Camera& camera, const glm::mat4& pose,
              int pose_id, bool writeToFile);

private:
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
};

#endif // RENDERER_H
