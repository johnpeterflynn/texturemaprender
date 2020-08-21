#include "keyhandler.h"

KeyHandler::KeyHandler() {

}

void KeyHandler::Subscribe(KeyListener &listener) {
    m_observers.push_back(&listener);
}

void KeyHandler::Unsubscribe(KeyListener &listener) {
    m_observers.remove(&listener);
}

void KeyHandler::ProcessKeystroke(GLFWwindow *window, float deltaTime)
{
    KeyListener::Key key;

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        key = KeyListener::Key::W;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        key = KeyListener::Key::S;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        key = KeyListener::Key::A;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        key = KeyListener::Key::D;
    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) {
        key = KeyListener::Key::SPACE;
    }
    if (glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS)
        key = KeyListener::Key::C;

    NotifyAll(key, deltaTime);
}

void KeyHandler::NotifyAll(KeyListener::Key key, float deltaTime) {
    for (auto & observer : m_observers) {
        observer->Notify(key, deltaTime);
    }
}