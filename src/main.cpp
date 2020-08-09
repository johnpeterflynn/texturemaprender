#include <glad/glad.h>
#include <GLFW/glfw3.h>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/string_cast.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <boost/program_options.hpp>

#include "cameraloader.h"
#include "frame_writer.h"

#include "camera.h"
#include "deferred_neural_renderer.h"

#include "renderer.h"
#include "scene.h"

namespace po = boost::program_options;

int run(std::string model_path, std::string poses_dir, string cam_params_dir, std::string output_path);
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void processInput(GLFWwindow *window);

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

DNRenderer dnr(RENDER_HEIGHT, RENDER_WIDTH);

FrameWriter frameWriter;
int num_snapshots = 0;

int main(int argc, char *argv[])
{
    // Declare required options.
    po::options_description required("Required options");
    required.add_options()
        ("model", po::value<std::string>(), "path to scan file (.off, .ply, etc..)")
        ("poses", po::value<std::string>(), "path to directory of camera poses (space separated .txt files)")
        ("net", po::value<std::string>(), "path to deferred neural renderer network model weights")
        ("cam-params", po::value<std::string>(), "path to directory containing intrinsic and extrinsic camera parameters")
        ("output-path", po::value<std::string>(), "path to output rendered frames")
    ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, required), vm);
    po::notify(vm);

    if (!vm.count("model")
            || !vm.count("poses")
            || !vm.count("cam-params")
            || !vm.count("output-path")) {
        std::cout << required << "\n";
        return 1;
    }

    // Init network model
    if (vm.count("net")) {
        dnr.load(vm["net"].as<std::string>());
    }

    int r = run(vm["model"].as<std::string>(),
                vm["poses"].as<std::string>(),
                vm["cam-params"].as<std::string>(),
                vm["output-path"].as<std::string>());

    return r;
}

int run(std::string model_path, std::string poses_dir,
        std::string cam_params_dir, std::string output_path)
{
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

    std::cout << "GL Version: " << glGetString(GL_VERSION) << "\n";

    // load models
    // -----------
    Scene scene(model_path);
    Renderer renderer(SCR_HEIGHT, SCR_WIDTH);

    // load camera poses, intrinsics and extrinsics
    // -----------------------------
    CameraLoader cam_loader(cam_params_dir, poses_dir);
    camera.setParams(cam_loader.m_intrinsics, cam_loader.m_extrinsics);

    frameWriter.setPath(output_path);

    // draw in wireframe
    //glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    // render loop
    // -----------

    while (!glfwWindowShouldClose(window))
    {
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
        renderer.draw(scene, camera);

        // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
        // -------------------------------------------------------------------------------
        glfwSwapBuffers(window);
        glfwPollEvents();

        // Write out framebuffers
        //if (!pose_processed) {
            //frameWriter.WriteAsTexcoord(num_processed_poses,
            //                            SCR_HEIGHT, SCR_WIDTH);
        //}

        if (!pose_processed) {
            pose_processed = true;
            num_processed_poses++;
        }
    }

    // glfw: terminate, clearing all previously allocated GLFW resources.
    // ------------------------------------------------------------------
    glfwTerminate();
    return 0;
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
    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) {
        frameWriter.RenderAsTexcoord(dnr, RENDER_HEIGHT, RENDER_WIDTH, true);
        //frameWriter.WriteAsJpg(num_snapshots++, SCR_HEIGHT, SCR_WIDTH);
    }
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
