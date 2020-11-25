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
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        NotifyAllKeys(KeyListener::Key::A, deltaTime);
    else
        NotifyKeyLifted(KeyListener::Key::A, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_B) == GLFW_PRESS)
        NotifyAllKeys(KeyListener::Key::B, deltaTime);
    else
        NotifyKeyLifted(KeyListener::Key::B, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS)
        NotifyAllKeys(KeyListener::Key::C, deltaTime);
    else
        NotifyKeyLifted(KeyListener::Key::C, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        NotifyAllKeys(KeyListener::Key::D, deltaTime);
    else
        NotifyKeyLifted(KeyListener::Key::D, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
        NotifyAllKeys(KeyListener::Key::E, deltaTime);
    else
        NotifyKeyLifted(KeyListener::Key::E, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_F) == GLFW_PRESS)
        NotifyAllKeys(KeyListener::Key::F, deltaTime);
    else
        NotifyKeyLifted(KeyListener::Key::F, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_G) == GLFW_PRESS)
        NotifyAllKeys(KeyListener::Key::G, deltaTime);
    else
        NotifyKeyLifted(KeyListener::Key::G, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_H) == GLFW_PRESS)
        NotifyAllKeys(KeyListener::Key::H, deltaTime);
    else
        NotifyKeyLifted(KeyListener::Key::H, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_I) == GLFW_PRESS)
        NotifyAllKeys(KeyListener::Key::I, deltaTime);
    else
        NotifyKeyLifted(KeyListener::Key::I, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_J) == GLFW_PRESS)
        NotifyAllKeys(KeyListener::Key::J, deltaTime);
    else
        NotifyKeyLifted(KeyListener::Key::J, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_K) == GLFW_PRESS)
        NotifyAllKeys(KeyListener::Key::K, deltaTime);
    else
        NotifyKeyLifted(KeyListener::Key::K, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_L) == GLFW_PRESS)
        NotifyAllKeys(KeyListener::Key::L, deltaTime);
    else
        NotifyKeyLifted(KeyListener::Key::L, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_M) == GLFW_PRESS)
        NotifyAllKeys(KeyListener::Key::M, deltaTime);
    else
        NotifyKeyLifted(KeyListener::Key::M, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_N) == GLFW_PRESS)
        NotifyAllKeys(KeyListener::Key::N, deltaTime);
    else
        NotifyKeyLifted(KeyListener::Key::N, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_O) == GLFW_PRESS)
        NotifyAllKeys(KeyListener::Key::O, deltaTime);
    else
        NotifyKeyLifted(KeyListener::Key::O, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_P) == GLFW_PRESS)
        NotifyAllKeys(KeyListener::Key::P, deltaTime);
    else
        NotifyKeyLifted(KeyListener::Key::P, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
        NotifyAllKeys(KeyListener::Key::Q, deltaTime);
    else
        NotifyKeyLifted(KeyListener::Key::Q, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS)
        NotifyAllKeys(KeyListener::Key::R, deltaTime);
    else
        NotifyKeyLifted(KeyListener::Key::R, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        NotifyAllKeys(KeyListener::Key::S, deltaTime);
    else
        NotifyKeyLifted(KeyListener::Key::S, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_T) == GLFW_PRESS)
        NotifyAllKeys(KeyListener::Key::T, deltaTime);
    else
        NotifyKeyLifted(KeyListener::Key::T, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_U) == GLFW_PRESS)
        NotifyAllKeys(KeyListener::Key::U, deltaTime);
    else
        NotifyKeyLifted(KeyListener::Key::U, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_V) == GLFW_PRESS)
        NotifyAllKeys(KeyListener::Key::V, deltaTime);
    else
        NotifyKeyLifted(KeyListener::Key::V, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        NotifyAllKeys(KeyListener::Key::W, deltaTime);
    else
        NotifyKeyLifted(KeyListener::Key::W, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_X) == GLFW_PRESS)
        NotifyAllKeys(KeyListener::Key::X, deltaTime);
    else
        NotifyKeyLifted(KeyListener::Key::X, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_Y) == GLFW_PRESS)
        NotifyAllKeys(KeyListener::Key::Y, deltaTime);
    else
        NotifyKeyLifted(KeyListener::Key::Y, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_Z) == GLFW_PRESS)
        NotifyAllKeys(KeyListener::Key::Z, deltaTime);
    else
        NotifyKeyLifted(KeyListener::Key::Z, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
        NotifyAllKeys(KeyListener::Key::SPACE, deltaTime);
    else
        NotifyKeyLifted(KeyListener::Key::SPACE, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
        NotifyAllKeys(KeyListener::Key::LEFT_SHIFT, deltaTime);
    else
        NotifyKeyLifted(KeyListener::Key::LEFT_SHIFT, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_MINUS) == GLFW_PRESS)
        NotifyAllKeys(KeyListener::Key::MINUS, deltaTime);
    else
        NotifyKeyLifted(KeyListener::Key::MINUS, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_EQUAL) == GLFW_PRESS)
        NotifyAllKeys(KeyListener::Key::EQUAL, deltaTime);
    else
        NotifyKeyLifted(KeyListener::Key::EQUAL, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_9) == GLFW_PRESS)
        NotifyAllKeys(KeyListener::Key::NINE, deltaTime);
    else
        NotifyKeyLifted(KeyListener::Key::NINE, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_0) == GLFW_PRESS)
        NotifyAllKeys(KeyListener::Key::ZERO, deltaTime);
    else
        NotifyKeyLifted(KeyListener::Key::ZERO, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_COMMA) == GLFW_PRESS)
        NotifyAllKeys(KeyListener::Key::COMMA, deltaTime);
    else
        NotifyKeyLifted(KeyListener::Key::COMMA, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_PERIOD) == GLFW_PRESS)
        NotifyAllKeys(KeyListener::Key::PERIOD, deltaTime);
    else
        NotifyKeyLifted(KeyListener::Key::PERIOD, deltaTime);

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
        observer->NotifyKeys(key, deltaTime, keyIsPressed(key));
    }
        m_key_pressed[key] = true;
}

void KeyHandler::NotifyKeyLifted(KeyListener::Key key, float deltaTime) {
    m_key_pressed[key] = false;
}

bool KeyHandler::keyIsPressed(KeyListener::Key key) {
    return m_key_pressed[key];
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
