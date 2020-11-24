#include <glad/glad.h>
#include <GLFW/glfw3.h>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/string_cast.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <boost/program_options.hpp>

#include <cuda_gl_interop.h>

#include "keyhandler.h"

#include "renderer.h"
#include "scene.h"

namespace po = boost::program_options;

int run(const Scene::Params &params,  int window_height, int window_width, std::string net_path,
        std::string output_path, bool write_coords, bool record_video);
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void processInput(GLFWwindow *window);

// settings
float c = 1;
const unsigned int SCR_DEFAULT_WIDTH = 1296;//686;//int(1296 * c);//686;
const unsigned int SCR_DEFAULT_HEIGHT = 968;//512;//nt(968 * c);//512;

// timing
float deltaTime = 0.0f;
float lastFrame = 0.0f;

// poses
const float POSES_PERIOD_SEC = 1.0f / 2.0f;
float lastPoseTime = 0.0f;
bool pose_processed = false;
glm::mat4 current_pose = glm::mat4(1.0f);

int num_snapshots = 0;

bool free_mode = true;

KeyHandler *key_handler;

int main(int argc, char *argv[])
{
    // Declare required options.
    po::options_description required("Required options");
    required.add_options()
        ("model", po::value<std::string>(), "path to scan file (.off, .ply, etc..)")
        ("agg-path", po::value<std::string>(), "path to aggregation file containing seg groups for each model vertex")
        ("segs-path", po::value<std::string>(), "path to segmentation file containing seg groups for each semantic object")
        ("poses", po::value<std::string>(), "path to directory of camera poses (space separated .txt files)")
        ("net", po::value<std::string>(), "path to deferred neural renderer network model weights")
        ("cam-params", po::value<std::string>(), "path to directory containing intrinsic and extrinsic camera parameters")
        ("output-path", po::value<std::string>(), "path to output rendered frames")
        ("write-coords", po::value<bool>()->default_value(false), "flag to write rendered texture coords to file")
        ("free-mode", po::value<bool>()->default_value(free_mode), "allow free moving camera")
        //("pose-start", po::value<int>()->default_value(0), "start pose index")
        ("record-video", po::value<bool>()->default_value(false), "record video on startup")
        ("height", po::value<int>()->default_value(SCR_DEFAULT_HEIGHT), "Window render height")
        ("width", po::value<int>()->default_value(SCR_DEFAULT_WIDTH), "Window render width")
        ("scene-mask", po::value<int>()->default_value(Model::DEFAULT_MASK), "ID for scene's associated texture map. Only needed for UV map generation")
        ("interp-factor", po::value<float>()->default_value(1.0), "Number of interpolated poses per provided pose. 1.0 means no interpolation.")
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

    free_mode = vm["free-mode"].as<bool>();
    //num_processed_poses = vm["pose-start"].as<int>();

    Scene::Params scene_params;
    scene_params.model_path = vm["model"].as<std::string>();
    scene_params.aggregation_path = vm["agg-path"].as<std::string>();
    scene_params.segs_path = vm["segs-path"].as<std::string>();
    scene_params.cam_params_dir = vm["cam-params"].as<std::string>();
    scene_params.poses_dir = vm["poses"].as<std::string>();
    scene_params.scene_mask = vm["scene-mask"].as<int>();
    scene_params.pose_interp_factor = vm["interp-factor"].as<float>();

    // TODO: Import these from one of the scene config files.
    scene_params.projection_height = 968;
    scene_params.projection_width = 1296;

    int r = run(scene_params,
                vm["height"].as<int>(),
                vm["width"].as<int>(),
                vm["net"].as<std::string>(),
                vm["output-path"].as<std::string>(),
                vm["write-coords"].as<bool>(),
                vm["record-video"].as<bool>());

    return r;
}

int run(const Scene::Params &scene_params, int window_height, int window_width, std::string net_path,
        std::string output_path, bool write_coords, bool record_video)
{

  int d;
    cudaError_t err = cudaGetDevice(&d);
      if (err != cudaSuccess) printf("kernel cuda error: %d, %s\n", (int)err, cudaGetErrorString(err));
        printf("device = %d\n", d);

    int code = cudaGLSetGLDevice(0);
    std::cout << "CUDA code: " << code << "\n";
    
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
    int glfw_window_height = window_height;
    int glfw_window_width = window_width;
    // Handle retina displays mucking with window resolution
#ifdef __APPLE__
    glfw_window_height /= 2;
    glfw_window_width /= 2;
#endif
    GLFWwindow* window = glfwCreateWindow(glfw_window_width, glfw_window_height, "uv-map Generator View", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    // TODO: Add this to the lifecycle of an object or larger scope
    key_handler = new KeyHandler(window_height, window_width);

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
    Scene scene(scene_params);
    Renderer renderer(window_height, window_width, net_path, output_path, record_video);

    key_handler->Subscribe(scene);
    key_handler->Subscribe(renderer);

    // draw in wireframe
    //glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    // render loop
    // -----------

    while (!glfwWindowShouldClose(window) && !scene.isFinished())
    {
        // per-frame time logic
        // --------------------
        float currentFrame = glfwGetTime();
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        // input
        // -----
        processInput(window);

        // scene
        // ------
        scene.Update(free_mode);

        // render
        // ------
        renderer.Draw(scene, write_coords);

        // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
        // -------------------------------------------------------------------------------
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    key_handler->Unsubscribe(scene);
    key_handler->Unsubscribe(renderer);

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

    key_handler->ProcessKeystroke(window, deltaTime);

    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) {
        //frameWriter.WriteAsJpg(num_snapshots++, SCR_HEIGHT, SCR_WIDTH);
    }

    // TODO: Move this elsehwere (into scene.cpp?)
    if (glfwGetKey(window, GLFW_KEY_B) == GLFW_PRESS) {
        free_mode = !free_mode;
    }
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width * 2, height * 2);
}

// glfw: whenever the mouse moves, this callback is called
// -------------------------------------------------------
void mouse_callback(GLFWwindow* window, double xpos, double ypos)
{
    key_handler->MouseCallback(window, xpos, ypos);
}

// glfw: whenever the mouse scroll wheel scrolls, this callback is called
// ----------------------------------------------------------------------
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    key_handler->ScrollCallback(window, xoffset, yoffset);
}
