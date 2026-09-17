#include "FMM_tSNE/cameras/two_D_camera.h"
#define GLM_ENABLE_EXPERIMENTAL
//  #define GLFW_INCLUDE_NONE // needed because glad and glfw both import the same thing that
//  conflicts

#include <glad/glad.h>
// glad must be first
// #include <FMM_tSNE/cameras/two_D_camera.h>
#include <FMM_tSNE/opengl_interaction/scene.h>
#include <GLFW/glfw3.h>
#include <filesystem>
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

void framebuffer_size_callback(GLFWwindow* window, int width, int height);

unsigned int screenWidth = 1920;
unsigned int screenHeight = 1080;

float deltaTime = 0.0f; // Time between current frame and last frame
float lastFrame = 0.0f; // Time of last frame

int frameCounter = 0;
int frameCounted = 0;

int per = 0;

int main()
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
    GLFWwindow* window = glfwCreateWindow(screenWidth, screenHeight, "master thesis", NULL, NULL);
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

    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    // glfwMaximizeWindow(window);
    // int xpos, ypos;
    // glfwGetMonitorPos(monitor, &xpos, &ypos);
    // glfwSetWindowPos(window, xpos, ypos);

    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSwapInterval(0); // unlimited frames!!!
    // glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    // glad: load all OpenGL function pointers
    // ---------------------------------------
    if (!gladLoadGLLoader((GLADloadproc) glfwGetProcAddress)) // load glad
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = nullptr;
    (void) io;
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    {
        // global stuff
        std::map<std::string, Scene*> scenes;
        std::string current_scene_name = "tsne";

        glm::mat4 model = glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, 0.0f)), glm::vec3(1.0f));

        Shader shader((std::filesystem::current_path().string() + "/shaders/shader_tsne.vs").c_str(),
                      (std::filesystem::current_path().string() + "/shaders/shader_tsne.fs").c_str());

        // create Buffer
        struct PosColStruct
        {
            glm::vec2 pos;
            glm::vec3 col;
        };
        std::vector<PosColStruct> pos_col_struct_vec = {
            {.pos = {20.0, 0.0}, .col = {1.0, 1.0, 1.0}},
            {.pos = {0.0, 20.0}, .col = {1.0, 1.0, 1.0}},
            {.pos = {0.0, 0.0}, .col = {1.0, 1.0, 1.0}},
        };
        Buffer buffer = Buffer(pos_col_struct_vec, DataLayout::Float2_Float3, GL_DYNAMIC_DRAW);

        Renderable renderable(GL_POINTS, model, &buffer, &shader, nullptr);
        std::vector<Renderable> renderables{renderable};

        TwoDCamera camera(glm::vec3(0.0f, 0.0f, -800.0f),
                          glm::vec3(0.0f, 1.0f, 0.0f),
                          90.0f,
                          0.0f,
                          glm::vec3(0.0f, 0.0f, -1.0f),
                          2.0f,
                          0.1f,
                          200.0f,
                          0.001f,
                          1000.0f,
                          false,
                          &screenWidth,
                          &screenHeight);

        Scene scene("tsne", &camera, renderables);

        scenes[scene.sceneName] = &scene;

        // scene names
        int current_scene_nameIndex = -1;
        std::vector<std::string> sceneNames;
        for (const std::pair<const std::string, Scene*>& scene : scenes)
        {
            sceneNames.push_back(scene.second->sceneName);
        }
        for (int i = 0; i < sceneNames.size(); i++)
        {
            if (sceneNames[i] == current_scene_name)
            {
                current_scene_nameIndex = i;
            }
        }

        // loop
        float lastTimePressed = 0.0f;
        float lastFrameUpdate = 0.0f;

        // render loop
        // -----------
        glEnable(GL_DEPTH_TEST);
        glPointSize(5.0f);
        while (!glfwWindowShouldClose(window))
        {
            // initial
            scenes[current_scene_name]->camera->processInput(window, deltaTime);
            glfwSetWindowUserPointer(window, scenes[current_scene_name]->camera);
            glfwSetCursorPosCallback(window, Camera::mouse_callback);
            glfwSetScrollCallback(window, Camera::scroll_callback);

            glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();

            float timeBeginFrame = glfwGetTime();
            // initial

            frameCounter++;
            if (glfwGetTime() - lastFrameUpdate > 1.0f)
            {
                lastFrameUpdate = glfwGetTime();
                frameCounted = frameCounter;
                frameCounter = 0;
            }

            ImGui::SetNextWindowSize(ImVec2(500, 250), ImGuiCond_FirstUseEver);
            ImGui::Begin("options");

            std::string frameOutput = "frames: " + std::to_string(frameCounted);
            ImGui::Text(frameOutput.c_str());

            current_scene_name = sceneNames[current_scene_nameIndex];
            ImGui::Combo(
                "Select scene",
                &current_scene_nameIndex,
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
                static_cast<void*>(&sceneNames),
                static_cast<int>(sceneNames.size()));
            // ImGui::Combo("Select scene", &sceneSelect, sceneNames,
            // IM_ARRAYSIZE(sceneNames.data()));
            //  this is kinda cursed, fix later!!!!!!!!!

            /*
            if (current_scene_name == "tsne")
            {
                if (per == 1)
                    scenes[current_scene_name]->camera->perspective = true;
                else
                    scenes[current_scene_name]->camera->perspective = false;

                std::string frameOutput = "iteration: " + std::to_string(tsne.iteration_counter);
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
                ImGui::SliderFloat("sim speed", &tsne.desired_iteration_per_second, 0.0f, 1000.0f);
                ImGui::SliderFloat("forceSize", &tsne.forceSize, 0.0f, 200.0f);
                ImGui::SliderInt("show tree level", &tsne.nodeLevelToShow, -1, 10);
                ImGui::SliderInt("follow embedded points", &tsne.follow, 0, 1);

                tsne.timeStep();

                if (tsne.follow == 1)
                {
                    // auto [left, right, down, up] = tsne.getEdges();
                    float left = tsne.minPos.x;
                    float down = tsne.minPos.y;
                    float right = tsne.maxPos.x;
                    float up = tsne.maxPos.y;
                    scenes[current_scene_name]->camera->Position =
                        glm::vec3(left + (right - left) * 0.5f,
                                  down + (up - down) * 0.5f,
                                  scenes[current_scene_name]->camera->Position.z);
                    scenes[current_scene_name]->camera->Zoom =
                        1.2f * std::max((up - down) * 0.5f, (right - left) * 0.5f);

                    // scenes[current_scene_name]->camera->Zoom = std::max(up - down, (right - left)
                    // / ((float)screenWidth / (float)screenHeight));
                }
            }
            else
            {
                std::cout << "no scene selected" << std::endl;
            }
    */

            scenes[current_scene_name]->Render();

            ImGui::End();
            ImGui::Render();
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

            // final
            glfwSwapBuffers(window);
            glfwPollEvents();
            float currentFrame = glfwGetTime();
            deltaTime = currentFrame - lastFrame;
            lastFrame = currentFrame;
            // final
        }

        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
    }

    glfwTerminate();
    std::cout << "hey\n";

    return 0;
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    screenWidth = width;
    screenHeight = height;
    glViewport(0, 0, screenWidth, screenHeight);
}
