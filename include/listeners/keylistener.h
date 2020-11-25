#ifndef KEYLISTENER_H
#define KEYLISTENER_H

class KeyListener {
public:
    enum class Key {
        A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X,
        Y, Z, SPACE, LEFT_SHIFT, MINUS, EQUAL
    };

    virtual void NotifyKeys(Key key, float deltaTime, bool is_already_pressed) = 0;
    virtual void NotifyMouse(double xoffset, double yoffset) {}
    virtual void NotifyScroll(double yoffset) {}
};

#endif // KEYLISTENER_H
