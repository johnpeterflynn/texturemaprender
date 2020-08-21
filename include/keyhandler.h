#ifndef KEYHANDLER_H
#define KEYHANDLER_H

#include <GLFW/glfw3.h>
#include <list>

#include "listeners/keylistener.h"

class KeyHandler {
public:
    KeyHandler();

    void Subscribe(KeyListener &listener);
    void Unsubscribe(KeyListener &listener);

    void ProcessKeystroke(GLFWwindow *window, float deltaTime);

private:
    void NotifyAll(KeyListener::Key key, float deltaTime);

    std::list<KeyListener*> m_observers;
};

#endif // KEYHANDLER_H
