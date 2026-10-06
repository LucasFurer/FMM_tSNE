#include <FMM_tSNE/app.h>
#include <FMM_tSNE/cameras/two_D_camera.h>
#include <FMM_tSNE/tsne/tsne_buffers.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

int main()
{
    App app{};

    // put stuff into the app
    glm::mat4 model = glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, 0.0f)), glm::vec3(1.0f));

    Shader shader = Shader((std::filesystem::current_path().string() + "/shaders/shader_tsne.vs").c_str(),
                           (std::filesystem::current_path().string() + "/shaders/shader_tsne.fs").c_str());

    TSNE_buffers tsne(1.0,            // min_theta
                      1.0,            // max_theta
                      1.0,            // cell_size
                      "MNIST_digits", // data_set: "MNIST_digits", "MNIST_fashion", "mice_brain_cells", "CIFAR10"
                      70000,          // data_size
                      30.0f,          // perplexity
                      216308u         // seed: 216308u, 592340823u, 4523u, 296343u
    );
    // tsne.nBodySelect = "FMM_SYM_MORTON";
    tsne.nBodySelect = "PM";

    Renderable renderable(GL_POINTS, model, tsne.embeddedBuffer, &shader, nullptr);
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
        app.run({.tsne = tsne});
    }

    return 0;
}
