#pragma once

#include <glad/glad.h>
// glad must be first
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <imgui/imgui.h>
#include <imgui/imgui_impl_glfw.h>
#include <imgui/imgui_impl_opengl3.h>

enum class Camera_Movement
{
    FORWARD,
    BACKWARD,
    LEFT,
    RIGHT,
    UP,
    DOWN
};

class Camera
{
  public:
    // camera Attributes
    glm::vec3 Position;
    glm::vec3 Front;
    glm::vec3 Up;
    glm::vec3 Right;
    glm::vec3 WorldUp;

    // euler Angles
    float Yaw;
    float Pitch;

    // camera options
    float MovementSpeed;
    float MouseSensitivity;
    float Zoom;
    float nearPlane;
    float farPlane;
    bool perspective;

    unsigned int* screenWidth;
    unsigned int* screenHeight;

    float lastX = 400.0f;
    float lastY = 300.0f;
    bool firstMouse = true;

    Camera(glm::vec3 initPosition,
           glm::vec3 up,
           float initYaw,
           float initPitch,
           glm::vec3 initFront,
           float initMovementSpeed,
           float initSensitivity,
           float initZoom,
           float initNearPlane,
           float initFarPlane,
           bool initPerspective,
           unsigned int* initScreenWidth,
           unsigned int* initScreenHeight);

    virtual ~Camera() = default;

    Camera(const Camera&) noexcept = default;
    Camera& operator=(const Camera&) noexcept = default;
    Camera(Camera&&) noexcept = default;
    Camera& operator=(Camera&&) noexcept = default;

    [[nodiscard]] glm::mat4 getViewMatrix() const;

    [[nodiscard]] glm::mat4 getProjectionMatrix();

    virtual void processInput(GLFWwindow* window, float deltaTime) = 0;

    static void mouse_callback(GLFWwindow* window, double xposIn, double yposIn);
    virtual void mouse_callback_impl(GLFWwindow* window, double xposIn, double yposIn) = 0;

    static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
    virtual void scroll_callback_impl(GLFWwindow* window, double xoffset, double yoffset) = 0;

    void updateCameraVectors();
};
