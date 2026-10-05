#include <FMM_tSNE/cameras/camera.h>

#include <GLFW/glfw3.h>
#include <algorithm>
#include <cmath>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <imgui/imgui.h>
#include <imgui/imgui_impl_glfw.h>
#include <imgui/imgui_impl_opengl3.h>

Camera::Camera(glm::vec3 initPosition,
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
               unsigned int* initScreenHeight)
    : Position(initPosition),
      Front(initFront),
      Up(0.0f, 1.0f, 0.0f),
      Right(1.0f, 0.0f, 0.0f),
      WorldUp(up),
      Yaw(initYaw),
      Pitch(initPitch),
      MovementSpeed(initMovementSpeed),
      MouseSensitivity(initSensitivity),
      Zoom(initZoom),
      nearPlane(initNearPlane),
      farPlane(initFarPlane),
      perspective(initPerspective),
      screenWidth(initScreenWidth),
      screenHeight(initScreenHeight)
{
    updateCameraVectors();
}

glm::mat4 Camera::getViewMatrix() const
{
    return glm::lookAt(Position, Position + Front, Up);
}

glm::mat4 Camera::getProjectionMatrix()
{
    const float aspect = static_cast<float>(*screenWidth) / static_cast<float>(*screenHeight);

    if (perspective)
    {
        Zoom = std::clamp(Zoom, 1.0f, 45.0f);
        return glm::perspective(glm::radians(Zoom), aspect, nearPlane, farPlane);
    }

    return glm::ortho(-Zoom * aspect, Zoom * aspect, -Zoom, Zoom, nearPlane, farPlane);
}

void Camera::mouse_callback(GLFWwindow* window, double xposIn, double yposIn)
{
    ImGui_ImplGlfw_CursorPosCallback(window, xposIn,
                                     yposIn); // needed so that imgui stays responsive
    auto* cam = static_cast<Camera*>(glfwGetWindowUserPointer(window));
    cam->mouse_callback_impl(window, xposIn, yposIn);
}

void Camera::scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    auto* cam = static_cast<Camera*>(glfwGetWindowUserPointer(window));
    cam->scroll_callback_impl(window, xoffset, yoffset);
}

void Camera::updateCameraVectors()
{
    const glm::vec3 front{std::cos(glm::radians(Yaw)) * std::cos(glm::radians(Pitch)),
                          std::sin(glm::radians(Pitch)),
                          std::sin(glm::radians(Yaw)) * std::cos(glm::radians(Pitch))};

    Front = glm::normalize(front);
    // normalize Right/Up because their length shrinks the closer Front gets to WorldUp,
    // which would otherwise slow movement down when looking straight up/down.
    Right = glm::normalize(glm::cross(Front, WorldUp));
    Up = glm::normalize(glm::cross(Right, Front));
}
