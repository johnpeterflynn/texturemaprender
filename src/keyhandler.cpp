#include "keyhandler.h"

KeyHandler::KeyHandler(int screen_height, int screen_width)
    : m_first_mouse(true)
    , m_lastX(screen_width / 2.0)
    , m_lastY(screen_height / 2.0)
{

}

void KeyHandler::Subscribe(KeyListener &listener) {
    m_observers.push_back(&listener);
}

void KeyHandler::Unsubscribe(KeyListener &listener) {
    m_observers.remove(&listener);
}

void KeyHandler::ProcessKeystroke(GLFWwindow *window, float deltaTime)
{
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        NotifyAllKeys(KeyListener::Key::W, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        NotifyAllKeys(KeyListener::Key::S, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        NotifyAllKeys(KeyListener::Key::A, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        NotifyAllKeys(KeyListener::Key::D, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
        NotifyAllKeys(KeyListener::Key::SPACE, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS)
        NotifyAllKeys(KeyListener::Key::C, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
        NotifyAllKeys(KeyListener::Key::LEFT_SHIFT, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_P) == GLFW_PRESS)
        NotifyAllKeys(KeyListener::Key::P, deltaTime);
}

// glfw: whenever the mouse moves, this callback is called
// -------------------------------------------------------
void KeyHandler::MouseCallback(GLFWwindow* window, double xpos, double ypos)
{
    if (m_first_mouse)
    {
        m_lastX = xpos;
        m_lastY = ypos;
        m_first_mouse = false;
    }

    float xoffset = xpos - m_lastX;
    float yoffset = m_lastY - ypos; // reversed since y-coordinates go from bottom to top

    m_lastX = xpos;
    m_lastY = ypos;

   NotifyAllMouse(xoffset, yoffset);
}

// glfw: whenever the mouse scroll wheel scrolls, this callback is called
// ----------------------------------------------------------------------
void KeyHandler::ScrollCallback(GLFWwindow* window, double xoffset, double yoffset)
{
    NotifyAllScroll(yoffset);
}


void KeyHandler::NotifyAllKeys(KeyListener::Key key, float deltaTime) {
    for (auto & observer : m_observers) {
        observer->NotifyKeys(key, deltaTime);
    }
}

void KeyHandler::NotifyAllMouse(double xoffset, double yoffset) {
    for (auto & observer : m_observers) {
        observer->NotifyMouse(xoffset, yoffset);
    }
}

void KeyHandler::NotifyAllScroll(double yoffset) {
    for (auto & observer : m_observers) {
        observer->NotifyScroll(yoffset);
    }
}