#ifndef KEYLISTENER_H
#define KEYLISTENER_H

class KeyListener {
public:
    enum class Key {
        A, C, D, S, W, SPACE
    };

    virtual void Notify(Key key, float deltaTime) = 0;
};

#endif // KEYLISTENER_H
