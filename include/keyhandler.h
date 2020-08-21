#ifndef KEYHANDLER_H
#define KEYHANDLER_H

#include <GLFW/glfw3.h>
#include <list>

#include "listeners/keylistener.h"

class KeyHandler {
public:
    KeyHandler(int height, int width);

    void Subscribe(KeyListener &listener);
    void Unsubscribe(KeyListener &listener);

    void ProcessKeystroke(GLFWwindow *window, float deltaTime);

    void MouseCallback(GLFWwindow* window, double xpos, double ypos);
    void ScrollCallback(GLFWwindow* window, double xoffset, double yoffset);

private:
    void NotifyAllKeys(KeyListener::Key key, float deltaTime);
    void NotifyAllMouse(double xoffset, double yoffset);
    void NotifyAllScroll(double yoffset);

    std::list<KeyListener*> m_observers;

    bool m_first_mouse;
    float m_lastX;
    float m_lastY;
};

#endif // KEYHANDLER_H
