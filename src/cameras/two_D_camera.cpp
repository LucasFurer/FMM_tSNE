#include <FMM_tSNE/cameras/two_D_camera.h>

void TwoDCamera::processInput(GLFWwindow* window, float deltaTime)
{
    const float velocity = MovementSpeed * Zoom * deltaTime;

    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) glfwSetWindowShouldClose(window, true);

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) Position += Up * velocity;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) Position -= Up * velocity;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) Position -= Right * velocity;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) Position += Right * velocity;
}

void TwoDCamera::mouse_callback_impl(GLFWwindow* /*window*/, double /*xposIn*/, double /*yposIn*/)
{
    // intentionally empty: TwoDCamera does not respond to mouse movement
}

void TwoDCamera::scroll_callback_impl(GLFWwindow* /*window*/, double /*xoffset*/, double yoffset)
{
    Zoom *= (-static_cast<float>(yoffset) / 20.0f) + 1.0f;
}
