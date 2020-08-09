#ifndef RENDERER_H
#define RENDERER_H

#include "camera.h"
#include "model.h"
#include "shader_s.h"

class Renderer {
public:
    Renderer(int height, int width);

    // TODO: Make params const
    void draw(Camera& camera, Model& model);

private:
    const int m_height, m_width;

    Shader m_uv_shader;
    Shader m_color_shader;

    unsigned int m_framebuffer;
    unsigned int m_texColorBuffer;
    unsigned int m_renderbuffer;
    unsigned int m_quadVAO, m_quadVBO;
};

#endif // RENDERER_H
