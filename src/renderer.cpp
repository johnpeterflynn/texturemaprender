#include "renderer.h"

#include <glad/glad.h>
#include <cuda_gl_interop.h>

Renderer::Renderer(int height, int width, const std::string &net_path,
                   const std::string &output_path)
    : m_height(height)
    , m_width(width)
    , m_uv_shader("src/shaders/vertexshader_texcoord.vs", "src/shaders/fragmentshader_texcoord.fs")
    , m_color_shader("src/shaders/vertexshader_vertcolor.vs", "src/shaders/fragmentshader_vertcolor.fs")
    , m_texture_shader("src/shaders/vertexshader_texture.vs", "src/shaders/fragmentshader_texture.fs")
    , m_frameWriter(output_path)
    , m_dnr(m_height, m_width, net_path)
    , m_b_snapshot(false)
    , m_b_recording_video(false)
{

    glGenFramebuffers(1, &m_framebuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, m_framebuffer);
    cudaGLRegisterBufferObject(m_framebuffer);
    
    // generate texture
    glGenTextures(1, &m_texColorBuffer);
    glBindTexture(GL_TEXTURE_2D, m_texColorBuffer);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, m_width, m_height, 0, GL_RG, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glBindTexture(GL_TEXTURE_2D, 0);

    // attach it to currently bound framebuffer object
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_texColorBuffer, 0);

    glGenRenderbuffers(1, &m_renderbuffer);
    glBindRenderbuffer(GL_RENDERBUFFER, m_renderbuffer);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, m_width, m_height);
    glBindRenderbuffer(GL_RENDERBUFFER, 0);

    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, m_renderbuffer);

    if(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE) {
        std::cout << "Framebuffer okay\n";
     }else {
        std::cout << "Framebuffer not okay\n";
    }
    
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    glGenBuffers(1, &m_cudabuffer);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_cudabuffer);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, m_width * m_height * 4,
		    NULL, GL_DYNAMIC_COPY);
    cudaGLRegisterBufferObject(m_cudabuffer);
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1,&m_cudatexColorBuffer);
    glBindTexture(GL_TEXTURE_2D, m_cudatexColorBuffer);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, m_width, m_height, 0, GL_BGRA,
		    GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glBindTexture(GL_TEXTURE_2D, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    
    float quadVertices[] = { // vertex attributes for a quad that fills the entire screen in Normalized Device Coordinates. NOTE that this plane is now much smaller and at the top of the screen
        // positions   // texCoords
        -1.0f,  1.0f,  0.0f, 1.0f,
        -1.0f,  -1.0f,  0.0f, 0.0f,
         1.0f,  -1.0f,  1.0f, 0.0f,

        -1.0f,  1.0f,  0.0f, 1.0f,
         1.0f,  -1.0f,  1.0f, 0.0f,
         1.0f,  1.0f,  1.0f, 1.0f
    };

    // screen quad VAO
    glGenVertexArrays(1, &m_quadVAO);
    glGenBuffers(1, &m_quadVBO);
    glBindVertexArray(m_quadVAO);
    glBindBuffer(GL_ARRAY_BUFFER, m_quadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));

    // configure global opengl state
    // -----------------------------
    glEnable(GL_DEPTH_TEST);
}

Renderer::~Renderer() {
    if (m_frameWriter.WriteVideoReady()) {
        m_frameWriter.ShutdownWriteVideo();
    }
}

void Renderer::Draw(Scene& scene, int pose_id, bool free_mode, bool writeToFile) {
    // render
    // ------
    glBindFramebuffer(GL_FRAMEBUFFER, m_framebuffer);
    glEnable(GL_DEPTH_TEST);
    // The depth (z component) of any screen space coordinates should be >0 for
    //  visible fragments. Make sure background color is set to 0.0 so that
    //  neural textures can ignore them
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // don't forget to enable shader before setting uniforms
    m_uv_shader.use();

    // view/projection transformations
    // TODO: Make m_camera private
    glm::mat4 projection = scene.m_camera.GetProjectionMatrix(m_height, m_width, 0.1f, 100.0f);

    // TODO: Resolve need to rotate the view by -90 and -180 degrees in these
    //  two modes.
    glm::mat4 view = glm::mat4(1.0f);
    if (free_mode) {
        // TODO: Make m_camera private
        view = scene.m_camera.GetViewMatrix()
                * glm::rotate(glm::mat4(1.0f), glm::radians(-90.0f),
                              glm::vec3(1.0f, 0.0f, 0.0f));
    }
    else {
        // TODO: Make m_camera_loader private
        // Rotate to make +Z the up direction as often defined by 3D scans
        view = glm::rotate(glm::mat4(1.0f), glm::radians(-180.0f),
                           glm::vec3(1.0f, 0.0f, 0.0f))
                * glm::inverse(scene.m_cam_loader.getPose(pose_id));
    }

    m_uv_shader.setMat4("projection", projection);
    m_uv_shader.setMat4("view", view);

    scene.Draw(m_uv_shader);

    if (writeToFile) {
        m_frameWriter.WriteAsTexcoord(pose_id, m_height, m_width);
    }

    m_frameWriter.RenderAsTexcoord(m_dnr, m_height, m_width, false);
    if (m_b_snapshot) {
        m_frameWriter.RenderAsTexcoord(m_dnr, m_height, m_width, true);
        m_b_snapshot = false;
    }

    uint8_t* dnr_out_gl;
    cudaGLMapBufferObject((void**)(&dnr_out_gl), m_cudabuffer);
    uint8_t* dnr_out_cuda = m_dnr.m_output.data_ptr<uint8_t>();
    // TODO: Pass pointers directly instead of copying data
    cudaMemcpy(dnr_out_gl, dnr_out_cuda, m_height * m_width * 3, cudaMemcpyDeviceToDevice);
    cudaGLUnmapBufferObject(m_cudabuffer);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_cudabuffer);
    glBindTexture(GL_TEXTURE_2D, m_cudatexColorBuffer);
    glTexSubImage2D(GL_TEXTURE_2D, 0 ,0, 0, m_width, m_height, GL_RGB, GL_UNSIGNED_BYTE, NULL);
    //glTexSubImage2D(GL_TEXTURE_2D, 0 ,0, 0, m_width, m_height, GL_RGB, GL_UNSIGNED_BYTE, (GLvoid*)data_out);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    
    // second pass
    glBindFramebuffer(GL_FRAMEBUFFER, 0); // back to default
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    m_texture_shader.use();
    glBindVertexArray(m_quadVAO);
    glDisable(GL_DEPTH_TEST);
    //glBindTexture(GL_TEXTURE_2D, m_texColorBuffer);
    glBindTexture(GL_TEXTURE_2D, m_cudatexColorBuffer);
    //uint8_t* data_out;
    //data_out = m_dnr.m_output.data_ptr<uint8_t>();

    //glTexSubImage2D(GL_TEXTURE_2D, 0 ,0, 0, m_width, m_height, GL_RGB, GL_UNSIGNED_BYTE, (GLvoid*)data_out);
    glDrawArrays(GL_TRIANGLES, 0, 6);

    if (m_b_recording_video && m_frameWriter.WriteVideoReady()) {
        m_frameWriter.WriteFrameAsVideo(m_height, m_width);
    }
}

void Renderer::NotifyKeys(Key key, float deltaTime) {
    switch(key) {
     case Key::P:
        m_b_snapshot = true;
        break;
     case Key::V:
        if (!m_b_recording_video) {
            m_frameWriter.SetupWriteVideo(m_height, m_width);
            m_b_recording_video = true;

            std::cout << "Starting video recording\n";
        }
        else {
            m_frameWriter.ShutdownWriteVideo();
            m_b_recording_video = false;

            std::cout << "Finishing video recording\n";
        }
        break;
    }
}
