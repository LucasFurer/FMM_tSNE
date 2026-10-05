// camera has controls build in, decouple this and make control functionality into app or something.
// i want the glfwSetWindowPointer or whatever to be in one place.
#pragma once

#include <FMM_tSNE/cameras/two_D_camera.h>
#include <FMM_tSNE/opengl_interaction/scene.h>
#include <FMM_tSNE/tsne/tsne_buffers.h>
#include <GLFW/glfw3.h>
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <imgui/imgui.h>
#include <imgui/imgui_impl_glfw.h>
#include <imgui/imgui_impl_opengl3.h>
#include <map>
#include <stb_image/stb_image.h>
#include <string>
#include <vector>

struct RunInfo
{
    TSNE_buffers& tsne;
};

class App
{
  public:
    App();
    ~App();

    App(const App& other) = delete;
    App& operator=(const App& other) = delete;
    App(const App&& other) = delete;
    App& operator=(const App&& other) = delete;

    [[nodiscard]] inline unsigned int get_screen_width_() const noexcept { return screen_width_; }
    [[nodiscard]] inline unsigned int get_screen_height_() const noexcept { return screen_height_; }

    void run(RunInfo run_info);
    bool should_close();

    // i dont like that this is static
    static void framebuffer_size_callback(GLFWwindow* window, int width, int height);

    // private:
    static inline unsigned int screen_width_{1920};
    static inline unsigned int screen_height_{1080};

    float delta_time_{0.0f}; // Time between current frame and last frame
    float last_frame_{0.0f}; // Time of last frame

    int frame_counter_{0};
    int frame_counted_{0}; // this weird data member is for keeping track of the frames per second

    int per_{0};

    GLFWwindow* window_{};

    std::map<std::string, Scene*> scenes_;
    std::string current_scene_name_ = "tsne";

    int current_scene_nameIndex_ = -1;
    std::vector<std::string> scene_names_;

    float last_time_pressed_ = 0.0f;
    float last_frame_update_ = 0.0f;
};
