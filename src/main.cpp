#include "FMM_tSNE/cameras/two_D_camera.h"
#define GLM_ENABLE_EXPERIMENTAL

#include <FMM_tSNE/app.h>

int main()
{
    App app{};

    // global stuff
    glm::mat4 model = glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, 0.0f)), glm::vec3(1.0f));

    Shader shader = Shader((std::filesystem::current_path().string() + "/shaders/shader_tsne.vs").c_str(),
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

    TwoDCamera camera = TwoDCamera(glm::vec3(0.0f, 0.0f, -800.0f),
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
                                   &App::screen_width_,
                                   &App::screen_height_);

    Scene scene = Scene("tsne", &camera, renderables);

    app.scenes_[scene.sceneName] = &scene;

    // scene names
    for (const std::pair<const std::string, Scene*>& scene : app.scenes_)
    {
        app.scene_names_.push_back(scene.second->sceneName);
    }
    for (int i = 0; i < app.scene_names_.size(); i++)
    {
        if (app.scene_names_[i] == app.current_scene_name_)
        {
            app.current_scene_nameIndex_ = i;
        }
    }

    while (!app.should_close())
    {
        app.run();
    }

    return 0;
}
