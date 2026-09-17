#pragma once

#include <FMM_tSNE/cameras/camera.h>

class TwoDCamera final : public Camera
{
  public:
    using Camera::Camera;

    void processInput(GLFWwindow* window, float deltaTime) override;
    void mouse_callback_impl(GLFWwindow* window, double xposIn, double yposIn) override;
    void scroll_callback_impl(GLFWwindow* window, double xoffset, double yoffset) override;
};
