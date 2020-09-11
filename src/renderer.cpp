#include "renderer.h"

#include <glad/glad.h>

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
    
    // generate texture
    glGenTextures(1, &m_texColorBuffer);
    glBindTexture(GL_TEXTURE_2D, m_texColorBuffer);
    // TODO: Use RGBA32F. RGB32F or RG32F?
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, m_width, m_height, 0, GL_RGBA, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glBindTexture(GL_TEXTURE_2D, 0);
    cudaGraphicsGLRegisterImage(&m_cgrs[0], m_texColorBuffer, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly);
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

    // CUDA to OpenGL
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1,&m_cudatexColorBuffer);
    glBindTexture(GL_TEXTURE_2D, m_cudatexColorBuffer);
    // TODO: Check appropriate data format
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, m_width, m_height, 0, GL_BGRA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glBindTexture(GL_TEXTURE_2D, 0);
    // Indicate that cuda will write to entire contents of texture but not read from it.
    cudaGraphicsGLRegisterImage(&m_cgrs[1], m_cudatexColorBuffer, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);

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

    // TODO: Group this with other cuda methods
    // Allocate cuda data
    cudaMalloc((void**)(&m_dnr_in_data_ptr), 4 * sizeof(float) * m_width * m_height);
}

Renderer::~Renderer() {    
    if (m_frameWriter.WriteVideoReady()) {
        m_frameWriter.ShutdownWriteVideo();
    }

    for (int i = 0; i < NUM_GRAPHICS_RESOURCES; i++) {
        cudaGraphicsUnregisterResource(m_cgrs[i]);
    }
    cudaFree(m_dnr_in_data_ptr);
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

    // OpenGL to CUDA
    // TODO: Map resources once for both directions
    cudaGraphicsMapResources(NUM_GRAPHICS_RESOURCES, m_cgrs);
    cudaArray* dnr_in_cuda_array;
    auto err_map = cudaGraphicsSubResourceGetMappedArray(&dnr_in_cuda_array, m_cgrs[0], 0, 0);
    // TODO: Define this in initialization
    int in_bytes_per_pixel = 4 * sizeof(float); // RGBA32F
    // TODO: Is this copy necessary?
    auto err_in = cudaMemcpy2DFromArray(m_dnr_in_data_ptr, in_bytes_per_pixel * m_width, dnr_in_cuda_array, 0, 0, in_bytes_per_pixel * m_width, m_height,
                                   cudaMemcpyDeviceToDevice);
    //std::cout << "Input error code: " << err_in << ": " << cudaGetErrorString(err_in) << "\n";
    m_dnr.render(m_dnr_in_data_ptr, m_height, m_width, false);

    // CUDA to OpenGl
    cudaArray* dnr_out_cuda_array;
    cudaGraphicsSubResourceGetMappedArray(&dnr_out_cuda_array, m_cgrs[1], 0, 0);
    uint8_t* dnr_out_data_ptr = m_dnr.m_output.data_ptr<uint8_t>();
    // TODO: Define this in initialization
    int out_bytes_per_pixel = 4 * sizeof(uint8_t); // RGBA8
    // TODO: Is this copy necessary?
    auto err_out = cudaMemcpy2DToArray(dnr_out_cuda_array, 0, 0, dnr_out_data_ptr, out_bytes_per_pixel * m_width, out_bytes_per_pixel * m_width, m_height,
                                   cudaMemcpyDeviceToDevice);
    //std::cout << "Output error code: " << err_out << ": " << cudaGetErrorString(err_out) << "\n";
    cudaGraphicsUnmapResources(NUM_GRAPHICS_RESOURCES, m_cgrs);

    if (m_b_snapshot) {
        m_frameWriter.RenderAsTexcoord(m_dnr, m_height, m_width, true);
        m_b_snapshot = false;
    }

    // second pass
    glBindFramebuffer(GL_FRAMEBUFFER, 0); // back to default
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    m_texture_shader.use();
    glBindVertexArray(m_quadVAO);
    glDisable(GL_DEPTH_TEST);
    glBindTexture(GL_TEXTURE_2D, m_cudatexColorBuffer);
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
            m_b_recording_video = m_frameWriter.SetupWriteVideo(m_height, m_width);
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
