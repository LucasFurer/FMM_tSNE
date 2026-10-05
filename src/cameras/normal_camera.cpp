#include <FMM_tSNE/cameras/normal_camera.h>

#include <GLFW/glfw3.h>
#include <algorithm>

void NormalCamera::processInput(GLFWwindow* window, float deltaTime)
{
    const float velocity = MovementSpeed * deltaTime;

    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) glfwSetWindowShouldClose(window, true);

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) Position += Front * velocity;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) Position -= Front * velocity;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) Position -= Right * velocity;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) Position += Right * velocity;
    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) Position += Up * velocity;
    if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS) Position -= Up * velocity;
}

void NormalCamera::mouse_callback_impl(GLFWwindow* window, double xposIn, double yposIn)
{
    const auto xpos = static_cast<float>(xposIn);
    const auto ypos = static_cast<float>(yposIn);

    if (firstMouse) // initially set to true
    {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos; // reversed since y-coordinates range from bottom to top
    lastX = xpos;
    lastY = ypos;

    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS)
    {
        xoffset *= MouseSensitivity;
        yoffset *= MouseSensitivity;

        Yaw += xoffset;
        // clamp pitch so the screen doesn't flip when looking straight up/down
        Pitch = std::clamp(Pitch + yoffset, -89.0f, 89.0f);

        updateCameraVectors();
    }
}

void NormalCamera::scroll_callback_impl(GLFWwindow* /*window*/, double xoffset, double yoffset)
{
    (void) xoffset;
    Zoom *= (-static_cast<float>(yoffset) / 20.0f) + 1.0f;
}
