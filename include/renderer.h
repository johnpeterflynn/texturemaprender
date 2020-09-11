#ifndef RENDERER_H
#define RENDERER_H

#include "camera.h"
#include "deferred_neural_renderer.h"
#include "frame_writer.h"
//#include "interfaces/iscene.h"
#include "listeners/keylistener.h"
#include "scene.h"
#include "shader_s.h"

#include <cuda_gl_interop.h>

class Renderer : public KeyListener {
public:
    Renderer(int height, int width, const std::string &net_path,
             const std::string &output_path);
    ~Renderer();

    // TODO: Make params const
    void Draw(Scene& scene, int pose_id, bool free_mode, bool writeToFile);

private:
    void NotifyKeys(Key key, float deltaTime);

    const int m_height, m_width;

    Shader m_uv_shader;
    Shader m_color_shader;
    Shader m_texture_shader;

    FrameWriter m_frameWriter;
    bool m_b_recording_video;

    DNRenderer m_dnr;

    unsigned int m_framebuffer;
    unsigned int m_texColorBuffer;
    unsigned int m_renderbuffer;
    unsigned int m_quadVAO, m_quadVBO;

    unsigned int m_cudabuffer;
    unsigned int m_cudatexColorBuffer;

    static const int NUM_GRAPHICS_RESOURCES = 2;
    cudaGraphicsResource_t m_cgrs[NUM_GRAPHICS_RESOURCES];

    float* m_dnr_in_data_ptr;

    bool m_b_snapshot;
};

#endif // RENDERER_H
