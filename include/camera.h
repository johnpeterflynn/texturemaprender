#ifndef CAMERA_H
#define CAMERA_H

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <vector>

#include "object.h"

// Defines several possible options for camera movement. Used as abstraction to stay away from window-system specific input methods
enum Camera_Movement {
    FORWARD,
    BACKWARD,
    LEFT,
    RIGHT,
    UP,
    DOWN,
    TURN_LEFT,
    TURN_RIGHT,
    TURN_UP,
    TURN_DOWN
};

// Default camera values
const float YAW         = -90.0f;
const float PITCH       =  0.0f;
const float SPEED       =  2.5f;
const float SENSITIVITY =  0.1f;
const float ZOOM        =  44.0f;//45.0f;

// An abstract camera class that processes input and calculates the corresponding Euler Angles, Vectors and Matrices for use in OpenGL
class Camera : public Object
{
public:
    // camera Attributes
    glm::vec3 Front;
    glm::vec3 Up;
    glm::vec3 Right;
    glm::vec3 WorldUp;

    // camera options
    float MovementSpeed;
    float MouseSensitivity;
    float Zoom;

    // constructor with vectors
    Camera(glm::vec3 position = glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f), float yaw = YAW, float pitch = PITCH)
        : Object(position, yaw, pitch)
        , Front(glm::vec3(0.0f, 0.0f, -1.0f))
        , MovementSpeed(SPEED)
        , MouseSensitivity(SENSITIVITY)
        , Zoom(ZOOM)
    {
        WorldUp = up;
        updateCameraVectors();
    }
    // constructor with scalar values
    Camera(float posX, float posY, float posZ, float upX, float upY, float upZ, float yaw, float pitch)
        : Object(glm::vec3(posX, posY, posZ), yaw, pitch)
        , Front(glm::vec3(0.0f, 0.0f, -1.0f))
        , MovementSpeed(SPEED)
        , MouseSensitivity(SENSITIVITY)
        , Zoom(ZOOM)
    {
        WorldUp = glm::vec3(upX, upY, upZ);
        updateCameraVectors();
    }

    void setParams(glm::mat4 intrinsics, glm::mat4 extrinsics) {
        m_intrinsics = intrinsics;
        m_extrinsics = extrinsics;
    }

    glm::mat4 GetProjectionMatrix(const float height, const float width,
                                  const float near, const float far)
    {
        float left = -width/2.0f;
        float right = width/2.0f;
        float bottom = -height/2.0f;
        float top = height/2.0f;
        float alpha = m_intrinsics[0][0];
        float beta = m_intrinsics[1][1];
        float x0 = 0.0f;//m_intrinsics[][]
        float y0 = 0.0f;//m_intrinsics[][]
        left = left * (near / alpha);
        right = right * (near / alpha);
        top = top * (near / beta);
        bottom = bottom * (near / beta);
        left = left - x0;
        right = right - x0;
        top = top - y0;
        bottom = bottom - y0;

        auto projection = glm::frustum(left, right, bottom, top, near, far);

        return projection;
    }

    // TODO: WARNING: This function is currently incompatible with GetProjectionMatrix().
    //  Need to investigate why.
    // returns the view matrix calculated using Euler Angles and the LookAt Matrix
    glm::mat4 GetViewMatrix()
    {
        glm::mat4 view = glm::lookAt(m_position, m_position + Front, Up);

        // TODO: Does extrinsics need to be inverted?
        view = m_extrinsics * view;

        return view;
    }

    // processes input received from any keyboard-like input system. Accepts input parameter in the form of camera defined ENUM (to abstract it from windowing systems)
    void ProcessKeyboard(Camera_Movement direction, float deltaTime)
    {
        float velocity = MovementSpeed * deltaTime;
        if (direction == FORWARD)
            m_position += Front * velocity;
        if (direction == BACKWARD)
            m_position -= Front * velocity;
        if (direction == LEFT)
            m_position -= Right * velocity;
        if (direction == RIGHT)
            m_position += Right * velocity;
        if (direction == UP)
            m_position += WorldUp * velocity;
        if (direction == DOWN)
            m_position -= WorldUp * velocity;
        if (direction == TURN_LEFT) {
            ProcessMovement(-velocity*10, 0);
        }
        if (direction == TURN_RIGHT) {
            ProcessMovement(velocity*10, 0);
        }
        if (direction == TURN_UP) {
            ProcessMovement(0, velocity*10);
        }
        if (direction == TURN_DOWN) {
            ProcessMovement(0, -velocity*10);
        }
    }

    // processes input received from a mouse input system. Expects the offset value in both the x and y direction.
    void ProcessMouseMovement(float xoffset, float yoffset, GLboolean constrainPitch = true)
    {
        xoffset *= MouseSensitivity;
        yoffset *= MouseSensitivity;

        ProcessMovement(xoffset, yoffset);
     }

    void ProcessMovement(float xoffset, float yoffset, GLboolean constrainPitch = true)
    {
        m_yaw   += xoffset;
        m_pitch += yoffset;

        // make sure that when pitch is out of bounds, screen doesn't get flipped
        if (constrainPitch)
        {
            if (m_pitch > 89.0f)
                m_pitch = 89.0f;
            if (m_pitch < -89.0f)
                m_pitch = -89.0f;
        }

        // update Front, Right and Up Vectors using the updated Euler angles
        updateCameraVectors();
    }

    // processes input received from a mouse scroll-wheel event. Only requires input on the vertical wheel-axis
    void ProcessMouseScroll(float yoffset)
    {
        Zoom -= (float)yoffset;
        if (Zoom < 1.0f)
            Zoom = 1.0f;
        if (Zoom > 45.0f)
            Zoom = 45.0f;
    }

private:
    // calculates the front vector from the Camera's (updated) Euler Angles
    void updateCameraVectors()
    {
        // calculate the new Front vector
        glm::vec3 front;
        front.x = cos(glm::radians(m_yaw)) * cos(glm::radians(m_pitch));
        front.y = sin(glm::radians(m_pitch));
        front.z = sin(glm::radians(m_yaw)) * cos(glm::radians(m_pitch));
        Front = glm::normalize(front);
        // also re-calculate the Right and Up vector
        Right = glm::normalize(glm::cross(Front, WorldUp));  // normalize the vectors, because their length gets closer to 0 the more you look up or down which results in slower movement.
        Up    = glm::normalize(glm::cross(Right, Front));
    }

    glm::mat4 m_intrinsics;
    glm::mat4 m_extrinsics;
};
#endif
