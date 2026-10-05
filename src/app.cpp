#include <FMM_tSNE/app.h>

#include <FMM_tSNE/cameras/two_D_camera.h>
#include <FMM_tSNE/opengl_interaction/scene.h>
#include <GLFW/glfw3.h>
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/string_cast.hpp>
#include <imgui/imgui.h>
#include <imgui/imgui_impl_glfw.h>
#include <imgui/imgui_impl_opengl3.h>
#include <iostream>
#include <map>
#include <stb_image/stb_image.h>
#include <string>
#include <vector>

App::App()
{
    //_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);

    // glfw: initialize and configure
    // ------------------------------
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    // glfw window creation
    // --------------------
    window_ = glfwCreateWindow(screen_width_, screen_height_, "master thesis", NULL, NULL);
    // GLFWmonitor* monitor = glfwGetPrimaryMonitor();
    // const GLFWvidmode* mode = glfwGetVideoMode(monitor);

    // Remove borders/title bar
    // glfwWindowHint(GLFW_DECORATED, GLFW_FALSE);

    // (Optional but good) prevent resizing artifacts
    // glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    // GLFWwindow* window = glfwCreateWindow(
    //     mode->width,
    //     mode->height,
    //     "master thesis",
    //     NULL,//monitor,   // <-- THIS enables fullscreen
    //     NULL
    //);

    if (window_ == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        // return -1;
    }
    // glfwMaximizeWindow(window);
    // int xpos, ypos;
    // glfwGetMonitorPos(monitor, &xpos, &ypos);
    // glfwSetWindowPos(window, xpos, ypos);

    glfwMakeContextCurrent(window_);
    glfwSetFramebufferSizeCallback(window_, framebuffer_size_callback);
    glfwSwapInterval(0); // unlimited frames!!!
    // glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    // glad: load all OpenGL function pointers
    // ---------------------------------------
    if (!gladLoadGLLoader((GLADloadproc) glfwGetProcAddress)) // load glad
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        // return -1;
    }

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = nullptr;
    (void) io;
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window_, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    {

        // loop
        // float last_time_pressed = 0.0f;
        // float last_frame_update = 0.0f;

        glEnable(GL_DEPTH_TEST);
        glPointSize(5.0f);
    }
}

App::~App()
{
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwTerminate();
}

void App::run(RunInfo run_info)
{
    // initial
    scenes_[current_scene_name_]->camera->processInput(window_, delta_time_);
    glfwSetWindowUserPointer(window_, scenes_[current_scene_name_]->camera);
    glfwSetCursorPosCallback(window_, Camera::mouse_callback);
    glfwSetScrollCallback(window_, Camera::scroll_callback);

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    float time_begin_frame = glfwGetTime();
    // initial

    frame_counter_++;
    if (glfwGetTime() - last_frame_update_ > 1.0f)
    {
        last_frame_update_ = glfwGetTime();
        frame_counted_ = frame_counter_;
        frame_counter_ = 0;
    }

    ImGui::SetNextWindowSize(ImVec2(500, 250), ImGuiCond_FirstUseEver);
    ImGui::Begin("options");

    std::string frame_output = "frames: " + std::to_string(frame_counted_);
    ImGui::Text(frame_output.c_str());

    current_scene_name_ = scene_names_[current_scene_nameIndex_];
    ImGui::Combo(
        "Select scene",
        &current_scene_nameIndex_,
        [](void* data, int idx, const char** out_text)
        {
            const std::vector<std::string>& vec = *static_cast<std::vector<std::string>*>(data);
            if (idx < 0 || idx >= vec.size())
            {
                return false;
            }
            *out_text = vec[idx].c_str();
            return true;
        },
        static_cast<void*>(&scene_names_),
        static_cast<int>(scene_names_.size()));
    // ImGui::Combo("Select scene", &sceneSelect, sceneNames,
    // IM_ARRAYSIZE(sceneNames.data()));
    //  this is kinda cursed, fix later!!!!!!!!!

    if (current_scene_name_ == "tsne")
    {
        scenes_[current_scene_name_]->camera->perspective = false;

        std::string frameOutput = "iteration: " + std::to_string(run_info.tsne.iteration_counter);
        ImGui::Text(frameOutput.c_str());

        // std::vector<std::string> solvers =
        // ImGui::Combo(
        //     "Select solver",
        //     &sceneSelect,
        //     [](void* data, int idx, const char** out_text)
        //     {
        //         const std::vector<std::string>& vec =
        //         *static_cast<std::vector<std::string>*>(data); if (idx < 0 || idx >=
        //         vec.size()) { return false; } *out_text = vec[idx].c_str(); return true;
        //     },
        //     static_cast<void*>(&sceneNames),
        //     static_cast<int>(sceneNames.size())
        //);
        ImGui::SliderFloat("sim speed", &run_info.tsne.desired_iteration_per_second, 0.0f, 1000.0f);
        ImGui::SliderFloat("forceSize", &run_info.tsne.forceSize, 0.0f, 200.0f);
        ImGui::SliderInt("show tree level", &run_info.tsne.nodeLevelToShow, -1, 10);
        ImGui::SliderInt("follow embedded points", &run_info.tsne.follow, 0, 1);

        run_info.tsne.timeStep();

        if (run_info.tsne.follow == 1)
        {
            // auto [left, right, down, up] = tsne.getEdges();
            float left = run_info.tsne.minPos.x;
            float down = run_info.tsne.minPos.y;
            float right = run_info.tsne.maxPos.x;
            float up = run_info.tsne.maxPos.y;
            scenes_[current_scene_name_]->camera->Position =
                glm::vec3(left + (right - left) * 0.5f,
                          down + (up - down) * 0.5f,
                          scenes_[current_scene_name_]->camera->Position.z);
            scenes_[current_scene_name_]->camera->Zoom = 1.2f * std::max((up - down) * 0.5f, (right - left) * 0.5f);

            // scenes[current_scene_name]->camera->Zoom = std::max(up - down, (right - left)
            // / ((float)screenWidth / (float)screenHeight));
        }
    }
    else
    {
        std::cout << "no scene selected" << std::endl;
    }

    scenes_[current_scene_name_]->Render();

    ImGui::End();
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    // final
    glfwSwapBuffers(window_);
    glfwPollEvents();
    float current_frame = glfwGetTime();
    delta_time_ = current_frame - last_frame_;
    last_frame_ = current_frame;
    // final
}

bool App::should_close()
{
    return static_cast<bool>(glfwWindowShouldClose(window_));
}

void App::framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    screen_width_ = width;
    screen_height_ = height;
    glViewport(0, 0, screen_width_, screen_height_);
}
