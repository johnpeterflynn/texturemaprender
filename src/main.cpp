#include <glad/glad.h>
#include <GLFW/glfw3.h>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/string_cast.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <boost/program_options.hpp>

#include "cameraloader.h"

#include "stb_image_write.h"

#include <iostream>
#include <fstream>

#include "shader_s.h"
#include "camera.h"
#include "model.h"
#include "deferred_neural_renderer.h"

namespace po = boost::program_options;

int run(std::string model_path, std::string poses_dir, std::string output_path);
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void processInput(GLFWwindow *window);
void writeFrameBuffer(std::string output_path, bool write);
void writeFrameBuffer_jpg(std::string output_path);

// settings
const unsigned int SCR_WIDTH = 1296;
const unsigned int SCR_HEIGHT = 968;

// camera
Camera camera(glm::vec3(0.0f, 0.0f, 0.0f));//3.0f));
float lastX = SCR_WIDTH / 2.0f;
float lastY = SCR_HEIGHT / 2.0f;
bool firstMouse = true;

// timing
float deltaTime = 0.0f;
float lastFrame = 0.0f;

// poses
const float POSES_PERIOD_SEC = 1.0f / 2.0f;
float lastPoseTime = 0.0f;
int num_processed_poses = 0;
bool pose_processed = false;
glm::mat4 current_pose = glm::mat4(1.0f);

// deferred neural renderer
int RENDER_HEIGHT = SCR_HEIGHT;//SCR_HEIGHT;//2*256;//SCR_HEIGHT;
int RENDER_WIDTH = SCR_WIDTH;//SCR_HEIGHT;//1296;//SCR_WIDTH;
bool livemode = true;
DNRenderer dnr(RENDER_HEIGHT, RENDER_WIDTH);

int main(int argc, char *argv[])
{
    // Declare required options.
    po::options_description required("Required options");
    required.add_options()
        ("model", po::value<std::string>(), "path to scan file (.off, .ply, etc..)")
        ("poses", po::value<std::string>(), "path to directory of camera poses (space separated .txt files)")
        ("net", po::value<std::string>(), "path to deferred neural renderer network model weights")
        ("output-path", po::value<std::string>(), "path to output rendered frames")
    ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, required), vm);
    po::notify(vm);

    if (!vm.count("model") || !vm.count("poses")|| !vm.count("output-path")) {
        std::cout << required << "\n";
        return 1;
    }

    // Init network model
    if (vm.count("net")) {
        dnr.load(vm["net"].as<std::string>());
    }

    int r = run(vm["model"].as<std::string>(),
                vm["poses"].as<std::string>(),
                vm["output-path"].as<std::string>());

    return r;
}

int run(std::string model_path, std::string poses_dir, std::string output_path) {
    // glfw: initialize and configure
    // ------------------------------
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    // glfw window creation
    // --------------------
    // TODO: Make sure this works on non-apple screens
    // see https://stackoverflow.com/questions/36672935/why-retina-screen-coordinate-value-is-twice-the-value-of-pixel-value
    int window_height = SCR_HEIGHT;
    int window_width = SCR_WIDTH;
    // Handle retina displays mucking with window resolution
#ifdef __APPLE__
    window_height /= 2;
    window_width /= 2;
#endif
    GLFWwindow* window = glfwCreateWindow(window_width, window_height, "uv-map Generator View", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);

    // tell GLFW to capture our mouse
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    // glad: load all OpenGL function pointers
    // ---------------------------------------
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    unsigned int framebuffer;
    glGenFramebuffers(1, &framebuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);

    // generate texture
    unsigned int texColorBuffer;
    glGenTextures(1, &texColorBuffer);
    glBindTexture(GL_TEXTURE_2D, texColorBuffer);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glBindTexture(GL_TEXTURE_2D, 0);

    // attach it to currently bound framebuffer object
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texColorBuffer, 0);

    unsigned int rbo;
    glGenRenderbuffers(1, &rbo);
    glBindRenderbuffer(GL_RENDERBUFFER, rbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, SCR_WIDTH, SCR_HEIGHT);
    glBindRenderbuffer(GL_RENDERBUFFER, 0);

    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rbo);

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
    unsigned int quadVAO, quadVBO;
    glGenVertexArrays(1, &quadVAO);
    glGenBuffers(1, &quadVBO);
    glBindVertexArray(quadVAO);
    glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));

    std::cout << "GL Version: " << glGetString(GL_VERSION) << "\n";

    // load camera poses, intrinsics and extrinsics
    // -----------------------------
    CameraLoader cam_loader(poses_dir);

    // tell stb_image.h to flip loaded texture's on the y-axis (before loading model).
    stbi_set_flip_vertically_on_load(false); // TODO: Why isn't this necessary?
    stbi_flip_vertically_on_write(false); // TODO: Why is this necessary?

    // configure global opengl state
    // -----------------------------
    glEnable(GL_DEPTH_TEST);

    // build and compile shaders
    // -------------------------
    Shader ourShader("src/vertexshader.vs", "src/fragmentshader.fs");
    Shader ourShaderFull("src/vertexshadercolor.vs", "src/fragmentshadercolor.fs");

    // load models
    // -----------
    Model ourModel(model_path);

    // draw in wireframe
    //glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    // render loop
    // -----------
    GLchar* data = new GLchar[SCR_HEIGHT * SCR_WIDTH * 3];
    for(int i = 0; i < SCR_HEIGHT * SCR_WIDTH * 3; i++) {
        data[i] = 255;
    }

    while (!glfwWindowShouldClose(window))
    {
        if (livemode) {
            glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
            glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // we're not using the stencil buffer now
            glEnable(GL_DEPTH_TEST);
        }

        //glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        // pose rate logic
        // --------------------
        //float currentPoseTime = glfwGetTime();
        //if (true || (currentPoseTime - lastPoseTime > POSES_PERIOD_SEC &&
        //        num_processed_poses < cam_loader.getNumPoses())) {
            pose_processed = false;
        //    lastPoseTime = currentPoseTime;
        //}


        // per-frame time logic
        // --------------------
        float currentFrame = glfwGetTime();
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        // input
        // -----
        processInput(window);

        // render
        // ------
        glClearColor(0.05f, 0.05f, 0.05f, 1.0f);
        //glClearColor(1.0f, 1.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // don't forget to enable shader before setting uniforms
        ourShader.use();

        // view/projection transformations
        glm::mat4 projection = glm::perspective(glm::radians(camera.Zoom), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);
        //std::cout << "Projection before: " << glm::to_string(projection) << "\n";
        float left = -(float)SCR_WIDTH/2.0f;//0.0f;
        float right = (float)SCR_WIDTH/2.0f;//(float)SCR_WIDTH;
        float bottom = -(float)SCR_HEIGHT/2.0f;//(float)SCR_HEIGHT;
        float top = (float)SCR_HEIGHT/2.0f;//0.0f;
        float near = 0.1f;
        float far = 100.0f;
        float alpha = 1169.621094f;
        float beta = 1167.105103f;
        float x0 = 0.0f;//646.295044f;
        float y0 = 0.0f;//489.927032f;
        left = left * (near / alpha);
        right = right * (near / alpha);
        top = top * (near / beta);
        bottom = bottom * (near / beta);
        left = left - x0;
        right = right - x0;
        top = top - y0;
        bottom = bottom - y0;

        projection = glm::frustum(left, right, bottom, top, near, far);
        //std::cout << "Projection after: " << glm::to_string(projection) << "\n";

        glm::mat4 view = camera.GetViewMatrix();
        //glm::mat4 view = glm::mat4(1.0f);
        // TODO: See below. Why can I rotate the view here but not the model later?
        view = glm::rotate(view, glm::radians(180.0f), glm::vec3(1.0f, 0.0f, 0.0f));

        ourShader.setMat4("projection", projection);
        ourShader.setMat4("view", view);

        // render the loaded model
        glm::mat4 model = glm::mat4(1.0f);
        if (!pose_processed) {
            //current_pose = cam_loader.getPose(num_processed_poses);
        }
        // TODO: Don't invert every time
        //glm::mat4 model = glm::inverse(current_pose);

        // TODO: Very strange. Why can't I rotate the model here?
        //model = glm::rotate(model, glm::radians(180.0f), glm::vec3(1.0f, 0.0f, 0.0f));

        //model = glm::translate(model, glm::vec3(0.0f, 0.0f, 0.0f)); // translate it down so it's at the center of the scene
        //model = glm::scale(model, glm::vec3(1.0f, 1.0f, 1.0f));	// it's a bit too big for our scene, so scale it down
        ourShader.setMat4("model", model);

        ourModel.Draw(ourShader);
        //glDrawBuffer(GL_BACK);
        //glDrawPixels(SCR_HEIGHT, SCR_WIDTH, GL_RGB, GL_FLOAT, data);
        //glClearColor(1.0f, 1.0f, 0.0f, 1.0f);

        //glLoadIdentity();
        //glRasterPos2i(0, 0);

        //glReadBuffer(GL_FRONT);
        //glReadPixels(0, 0, SCR_WIDTH, SCR_HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, data);

        if (livemode) {
            writeFrameBuffer("", false);

            // second pass
            glBindFramebuffer(GL_FRAMEBUFFER, 0); // back to default
            glClearColor(1.0f, 1.0f, 0.0f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT);

            ourShaderFull.use();
            glBindVertexArray(quadVAO);
            glDisable(GL_DEPTH_TEST);
            glBindTexture(GL_TEXTURE_2D, texColorBuffer);
            uint8_t* data_out = dnr.m_output.data_ptr<uint8_t>();

            glTexSubImage2D(GL_TEXTURE_2D, 0 ,0, 0, RENDER_WIDTH, RENDER_HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, (GLvoid*)data_out);
            glDrawArrays(GL_TRIANGLES, 0, 6);
        }




        // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
        // -------------------------------------------------------------------------------
        glfwSwapBuffers(window);
        glfwPollEvents();

        // Write out framebuffers
        if (!pose_processed) {
            //writeFrameBuffer(output_path);
        }

        if (!pose_processed) {
            pose_processed = true;
            num_processed_poses++;
        }
    }
    delete[] data;

    // glfw: terminate, clearing all previously allocated GLFW resources.
    // ------------------------------------------------------------------
    glfwTerminate();
    return 0;
}

 void writeFrameBuffer(std::string output_path, bool write)
{
    std::cout << "Start neural rendering\n";
    // TODO: Allocate and deallocate heap_data only once
    float *heap_data = new float[SCR_HEIGHT * SCR_WIDTH * 2];
    //GLchar *heap_data = new GLchar[SCR_HEIGHT * SCR_WIDTH * 3];

    glReadBuffer(GL_FRONT);
    glReadPixels(0, 0, SCR_WIDTH, SCR_HEIGHT, GL_RG, GL_FLOAT, heap_data);
    //glReadPixels(0, 0, SCR_WIDTH, SCR_HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, heap_data);

    //glBindFramebuffer(GL_FRAMEBUFFER, 0);

    //glDrawBuffer(GL_FRONT);

    dnr.render(heap_data, SCR_HEIGHT, SCR_WIDTH, write);





    // Maybe clear also

    //glLoadIdentity();
    //glRasterPos2i(0, 0);
    //glDrawPixels( RENDER_WIDTH, RENDER_HEIGHT, GL_RGB, GL_UNSIGNED_INT, heap_data);

    std::cout << "Finish neural rendering\n";

    //std::string filename = std::string("./output/") + to_string(num_processed_poses) + ".jpg";
    //std::string filename = output_path + "/" + to_string(num_processed_poses);
    //auto myfile = std::fstream(filename, std::ios::out | std::ios::binary);
    // Each pixel has a (u,v) coodrinate and each coordinate is a 4-byte float
    //myfile.write((char*)heap_data, SCR_HEIGHT * SCR_WIDTH * 2 * 4);
    //myfile.close();
    //std::cout << "Writing\n";
    //stbi_write_jpg(filename.c_str(), SCR_WIDTH, SCR_HEIGHT, 3, heap_data, 100);

    delete [] heap_data;
}

 void writeFrameBuffer_jpg(std::string output_path) {
     GLchar data[SCR_HEIGHT * SCR_WIDTH * 3]; // # pixels x # floats per pixel
      glReadBuffer(GL_FRONT);
     // TODO: Should this be GL_RGBA with 4 positions per pixel?
     glReadPixels(0, 0, SCR_WIDTH, SCR_HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, data);
     std::string filename = output_path + "/" + to_string(num_processed_poses) + ".jpg";

     // 100% quality, could be less
     stbi_write_jpg(filename.c_str(), SCR_WIDTH, SCR_HEIGHT, 3, data, 90);
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow *window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        camera.ProcessKeyboard(FORWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        camera.ProcessKeyboard(BACKWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        camera.ProcessKeyboard(LEFT, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        camera.ProcessKeyboard(RIGHT, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
        writeFrameBuffer("", true);
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}

// glfw: whenever the mouse moves, this callback is called
// -------------------------------------------------------
void mouse_callback(GLFWwindow* window, double xpos, double ypos)
{
    if (firstMouse)
    {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top

    lastX = xpos;
    lastY = ypos;

    camera.ProcessMouseMovement(xoffset, yoffset);
}

// glfw: whenever the mouse scroll wheel scrolls, this callback is called
// ----------------------------------------------------------------------
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    camera.ProcessMouseScroll(yoffset);
}
