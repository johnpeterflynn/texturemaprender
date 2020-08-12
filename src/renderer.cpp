#include "renderer.h"

#include <glad/glad.h>

Renderer::Renderer(int height, int width, const std::string &net_path,
                   const std::string &output_path)
    : m_height(height)
    , m_width(width)
    , m_uv_shader("src/vertexshader.vs", "src/fragmentshader.fs")
    , m_color_shader("src/vertexshadercolor.vs", "src/fragmentshadercolor.fs")
    , m_frameWriter(output_path)
    , m_dnr(m_height, m_width, net_path)
{
    glGenFramebuffers(1, &m_framebuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, m_framebuffer);

    // generate texture
    glGenTextures(1, &m_texColorBuffer);
    glBindTexture(GL_TEXTURE_2D, m_texColorBuffer);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, m_width, m_height, 0, GL_RGB, GL_FLOAT, NULL);
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

void Renderer::Draw(IScene& scene, Camera& camera, int pose_id, bool writeToFile) {
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
    glm::mat4 projection = camera.GetProjectionMatrix(m_height, m_width, 0.1f, 100.0f);
    //glm::mat4 projection = scene.GetProjectionMatrix();

    // TODO: Use view matrix from camera
    glm::mat4 view = camera.GetViewMatrix();
    //glm::mat4 view = scene.GetViewMatrix();
    //glm::mat4 view = glm::mat4(1.0f);

    m_uv_shader.setMat4("projection", projection);
    m_uv_shader.setMat4("view", view);

    // render the loaded model
    glm::mat4 model = scene.GetModelMatrix();

    //if (!pose_processed) {
        //current_pose = cam_loader.getPose(num_processed_poses);
    //}
    // TODO: Don't invert every time
    //glm::mat4 model = glm::inverse(current_pose);

    m_uv_shader.setMat4("model", model);

    scene.Draw(m_uv_shader);

    if (writeToFile) {
        m_frameWriter.WriteAsTexcoord(pose_id, m_height, m_width);
    }

    // second pass
    glBindFramebuffer(GL_FRAMEBUFFER, 0); // back to default
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    m_color_shader.use();
    glBindVertexArray(m_quadVAO);
    glDisable(GL_DEPTH_TEST);
    glBindTexture(GL_TEXTURE_2D, m_texColorBuffer);
    //uint8_t* data_out = dnr.m_output.data_ptr<uint8_t>();

    //glTexSubImage2D(GL_TEXTURE_2D, 0 ,0, 0, RENDER_WIDTH, RENDER_HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, (GLvoid*)data_out);
    glDrawArrays(GL_TRIANGLES, 0, 6);
}