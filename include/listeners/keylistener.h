#ifndef KEYLISTENER_H
#define KEYLISTENER_H

class KeyListener {
public:
    enum class Key {
        A, C, D, S, W, SPACE
    };

    virtual void NotifyKeys(Key key, float deltaTime) = 0;
    virtual void NotifyMouse(double xoffset, double yoffset) {}
    virtual void NotifyScroll(double yoffset) {}
};

#endif // KEYLISTENER_H
